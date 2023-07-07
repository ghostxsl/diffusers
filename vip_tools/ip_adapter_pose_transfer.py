# Copyright (c) wilson.xu. All rights reserved.
import os
import random
from os.path import join, exists, splitext, basename
import argparse
import time
from tqdm import tqdm
import torch
import numpy as np
from PIL import Image

from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import EulerAncestralDiscreteScheduler, KDPMPP2MDiscreteScheduler
from transformers import AutoTokenizer, CLIPTextModel, CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers.models.controlnetxs import ControlNetXSModel
from diffusers.pipelines.vip import VIPIPAPoseTransferPipeline
from diffusers.utils.prompt_parser import load_webui_textual_inversion

from diffusers.utils.vip_utils import *
from diffusers.data.utils import *
from diffusers.data.vos_client import VOSClient

from aistudio.utils.loader import ROOT_DIR
from aistudio.extensions.HumanPose import HumanPose
from aistudio.extensions.HumanPose.utils import POSE_CONFIG_DIR
from aistudio.extensions.CodeFormer import CodeFormer

# torch.backends.cuda.matmul.allow_tf32 = True

gender_map = {
        "男性": "man",
        "女性": "woman",
        "男童": "boy",
        "女童": "girl",
        "man": "man",
        "woman": "woman",
        "boy": "boy",
        "girl": "girl",
}
race_map = {
    "亚洲": "asian",
    "欧洲": "european",
    "欧美": "european",
    "asian": "asian",
    "european": "european",
}


def parse_args():
    parser = argparse.ArgumentParser(description="inference script.")
    parser.add_argument(
        "--ref_img",
        default=None,
        type=str,
        help="Path to reference image.")
    parser.add_argument(
        "--ref_dir",
        default=None,
        type=str,
        help="Directory to reference image.")
    parser.add_argument(
        "--vos_pkl",
        default=None,
        type=str,
        help="Path to image list on vos.")

    parser.add_argument(
        "--pose_dir",
        default=None,
        type=str,
        help="Directory to pose image.")
    parser.add_argument(
        "--pose_lib",
        default=None,
        type=str,
        help="Path to pose lib.")
    parser.add_argument(
        "--out_dir",
        default="output",
        type=str,
        help="Directory to save.")
    parser.add_argument(
        "--topk",
        default=2,
        type=int,
        help="")
    parser.add_argument(
        "--topk_thr",
        default=0.93,
        type=float,
        help="")
    parser.add_argument(
        "--topk_stride",
        default=None,
        type=float,
        help="")
    parser.add_argument(
        "--display",
        default=False,
        action="store_true",
        help="")

    parser.add_argument(
        "--rank",
        default=None,
        type=int)
    parser.add_argument(
        "--num_ranks",
        default=None,
        type=int)

    parser.add_argument(
        "--resolution",
        type=str,
        default="1024",
        help=(
            "The resolution(h x w) for input images, all the images"
            " will be resized to this resolution."
        ),
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="/apps/dat/cv/xsl/weights/majicMIX-realistic-7",
    )
    parser.add_argument(
        "--ip_adapter_model_path",
        type=str,
        default="/apps/dat/cv/xsl/exp_animate/checkpoint-55000/ema/ip_adapter_plus.bin",
    )
    parser.add_argument(
        "--vae_model_path",
        type=str,
        default="/apps/dat/cv/xsl/weights/vae_ft_mse",
    )
    parser.add_argument(
        "--image_encoder_model_path",
        type=str,
        default="/apps/dat/cv/xsl/weights/IP-Adapter/image_encoder",
    )
    parser.add_argument(
        "--controlnet_model_path",
        type=str,
        default="/apps/dat/cv/xsl/weights/control_v11p_sd15_openpose",
    )
    parser.add_argument(
        "--weight_dir",
        default=None,
        type=str,
        help="Directory to weights.")

    parser.add_argument(
        "--device",
        default='cuda',
        type=str,
        help="Device to use (e.g. cpu, cuda:0, cuda:1, etc.)")
    parser.add_argument(
        "--dtype",
        default='fp16',
        type=str,
        help="Data type to use (e.g. fp16, fp32, etc.)")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if len(args.resolution.split("x")) == 1:
        args.resolution = [int(args.resolution), ] * 2
    elif len(args.resolution.split("x")) == 2:
        args.resolution = [int(r) for r in args.resolution.split("x")]
    else:
        raise Exception(f"Error `resolution` type({type(args.resolution)}): {args.resolution}.")

    args.device = torch.device(args.device)
    if args.dtype == 'fp16':
        args.dtype = torch.float16
    else:
        args.dtype = torch.float32

    return args


class PoseTransfer(object):
    def __init__(self, args, crop_size=(768, 576), draw_hand=True, ip_adapter_scale=0.9):
        self.args = args
        self.crop_size = crop_size
        self.draw_hand = draw_hand
        self.ip_adapter_scale = ip_adapter_scale

        self.topk_thr = args.topk_thr
        self.topk_stride = args.topk_stride
        self.infer_size = args.resolution

        self.weight_dir = args.weight_dir or join(ROOT_DIR, "weights")
        self.device = args.device
        self.dtype = args.dtype
        if args.pose_lib:
            self.pose_lib = load_file(args.pose_lib)
        else:
            self.pose_lib = self.load_pose_lib(args.pose_dir)

        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model_path,
            subfolder="tokenizer",
            use_fast=False,
        )
        # import correct text encoder class
        text_encoder = CLIPTextModel.from_pretrained(
            args.base_model_path, subfolder="text_encoder").to(self.device, dtype=self.dtype)
        vae = AutoencoderKL.from_pretrained(args.vae_model_path).to(self.device, dtype=self.dtype)
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            args.image_encoder_model_path).to(self.device, dtype=self.dtype)

        unet = UNet2DConditionModel.from_pretrained(
            args.base_model_path, subfolder="unet")
        unet._init_ip_adapter_plus(state_dict=torch.load(args.ip_adapter_model_path, map_location=torch.device("cpu")))
        unet = unet.to(self.device, dtype=self.dtype)
        controlnet = ControlNetXSModel.from_pretrained(
            args.controlnet_model_path).to(self.device, dtype=self.dtype)

        scheduler = KDPMPP2MDiscreteScheduler.from_pretrained(
            args.base_model_path, subfolder="scheduler")
        self.pipe_pose_trans = VIPIPAPoseTransferPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            controlnet=controlnet,
            feature_extractor=CLIPImageProcessor(),
            image_encoder=image_encoder).to(self.device)

        # ip_adapter_scale = {
        #     "down": {"block_2": [1.0, 0.85]},
        #     "up": {"block_1": [1.0, 1.0, 1.0]},
        # }
        self.pipe_pose_trans.set_ip_adapter_scale(ip_adapter_scale)
        # self.pipe_pose_trans.enable_freeu(s1=0.9, s2=0.2, b1=1.5, b2=1.6)
        # load_webui_textual_inversion(join(self.weight_dir, "embedding"), self.pipe_pose_trans)

        self.pose_infer = HumanPose(
            det_cfg=join(POSE_CONFIG_DIR, "rtmdet_l_8xb32-300e_coco.py"),
            det_pth=join(self.weight_dir, "extensions/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth"),
            bodypose_cfg=join(POSE_CONFIG_DIR, "rtmpose-l_8xb256-420e_body8-256x192.py"),
            bodypose_pth=join(self.weight_dir,
                              "extensions/rtmpose-l_simcc-body7_pt-body7_420e-256x192-4dba18fc_20230504.pth"),
            wholebodypose_cfg=join(POSE_CONFIG_DIR, "dwpose_l_wholebody_384x288.py"),
            wholebodypose_pth=join(self.weight_dir, "extensions/dw-ll_ucoco_384.pth"),
            device=self.device,
            bbox_thr=0.2,
        )
        self.codeformer = CodeFormer(
            model_path=join(self.weight_dir, "extensions/codeformer-v0.1.0.pth"),
            det_model_path=join(self.weight_dir, "extensions/detection_Resnet50_Final.pth"),
            parsing_model_path=join(self.weight_dir, "extensions/parsing_parsenet.pth"),
            torch_compile=False,
            device=self.device,
        )

    def normalize_pose(self, pose, crop_size):
        ch, cw = crop_size
        w, h = pose['width'], pose['height']
        bboxes = pose['body']['bboxes']
        if len(bboxes) == 0:
            return None, None

        # 取得分最高的bbox, 变换为原始坐标
        bbox = bboxes[0] * np.array([w, h, w, h])
        # 取得分最高的keypoints, 变换为原始坐标
        body_kpts = pose['body']['keypoints'][0] * np.array([w, h, 1.])
        # 平移关键点
        body_kpts[..., :2] -= bbox[:2]

        # 按比例resize
        bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
        ratio_h, ratio_w = ch / bh, cw / bw

        out_ratio = ratio_h
        pad_shift = [0, 0] # x, y
        if ratio_h < ratio_w:
            # 按高 resize
            ow = int(bh / ch * cw)
            expand_w = ow - bw
            body_kpts[..., 0] += int(expand_w / 2)
            body_kpts[..., :2] *= ratio_h

            pad_shift[0] = int(expand_w / 2)
        elif ratio_h > ratio_w:
            # 按宽 resize
            oh = int(bw / cw * ch)
            expand_h = oh - bh
            body_kpts[..., 1] += int(expand_h / 2)
            body_kpts[..., :2] *= ratio_w

            out_ratio = ratio_w
            pad_shift[1] = int(expand_h / 2)
        else:
            body_kpts[..., :2] *= ratio_h

        return body_kpts[None], (out_ratio, pad_shift, bbox)

    def load_pose_lib(self, pose_dir):
        print("Load and normalize pose lib...")
        out = []
        for name in tqdm(os.listdir(pose_dir)):
            try:
                pose = load_file(join(pose_dir, name))
                body_kpts, _ = self.normalize_pose(pose, self.crop_size)
                if body_kpts is None:
                    continue

                out.append(body_kpts)
            except:
                continue
        return {
            "dts": np.concatenate(out)
        }

    def pad_image(self, img, pad_values=255):
        w, h = img.size
        img = np.array(img)
        pad_border = None
        if w > h:
            pad_ = w - h
            pad_border = ((pad_ // 2, pad_ - pad_ // 2), (0, 0), (0, 0))
            img = np.pad(
                img,
                pad_border,
                constant_values=pad_values
            )
        elif h > w:
            pad_ = h - w
            pad_border = ((0, 0), (pad_ // 2, pad_ - pad_ // 2), (0, 0))
            img = np.pad(
                img,
                pad_border,
                constant_values=pad_values
            )
        return Image.fromarray(img), pad_border

    def unpad_image(self, img, pad_border=None):
        if pad_border is not None:
            if sum(pad_border[0]) != 0:
                y1, y2 = pad_border[0]
                y2 = -y2 if y2 != 0 else img.height
                img = np.array(img)[y1: y2]
            elif sum(pad_border[1]) != 0:
                x1, x2 = pad_border[1]
                x2 = -x2 if x2 != 0 else img.width
                img = np.array(img)[:, x1: x2]
            else:
                raise Exception(f"Error: pad_border: {pad_border}")
            img = Image.fromarray(img)

        return img

    def crop_image(self, image, crop_size=(1024, 768)):
        image = np.array(image)
        bboxes, _ = self.pose_infer.detector(image)
        if len(bboxes) == 0:
            return None, None, None

        # 取得分最高的bbox
        det_bbox = np.int32(bboxes[0])
        crop_bbox, pad_ = crop_human_bbox(det_bbox, image.shape[:2], crop_size)
        x1, y1, x2, y2 = crop_bbox
        crop_img = image[y1: y2, x1: x2]
        if pad_ is not None:
            crop_img = np.pad(crop_img, pad_, constant_values=255)
        crop_img = Image.fromarray(crop_img).resize(crop_size[::-1], Image.LANCZOS)

        return crop_img, crop_bbox, pad_

    def crop_human_image(self, img, pose, pad_bbox=0):
        w, h = pose['width'], pose['height']
        bbox = np.int32(pose['body']['bboxes'][0] * np.array([w, h, w, h]))

        x1, y1, x2, y2 = bbox
        x1 = min(max(x1, 0), w)
        x2 = min(max(x2, 0), w)
        y1 = min(max(y1, 0), h)
        y2 = min(max(y2, 0), h)
        if pad_bbox > 0:
            x1 = x1 - pad_bbox if x1 - pad_bbox > 0 else 0
            y1 = y1 - pad_bbox if y1 - pad_bbox > 0 else 0
            x2 = x2 + pad_bbox if x2 + pad_bbox < w else w
            y2 = y2 + pad_bbox if y2 + pad_bbox < h else h

        img = np.array(img)
        crop_img = img[y1: y2, x1: x2]
        return Image.fromarray(crop_img)

    def draw_pose(self, body_pose, draw_size, hand_pose=None, face_pose=None, kpt_thr=0.3):
        height, width = draw_size
        canvas = np.zeros(shape=(height, width, 3), dtype=np.uint8)
        canvas = draw_bodypose(canvas, body_pose[..., :2], body_pose[..., 2] > kpt_thr)

        if hand_pose is not None:
            canvas = draw_handpose(canvas, hand_pose[..., :2], hand_pose[..., 2] > kpt_thr)
        if face_pose is not None:
            canvas = draw_facepose(canvas, face_pose[..., :2], face_pose[..., 2] > kpt_thr)

        return Image.fromarray(canvas)

    def topk_sampling(self, oks, topk=1):
        topk_ind = np.argsort(-oks)
        out_ind, out_val = [], []
        if self.topk_stride is not None:
            sroted_oks = oks[topk_ind]
            stride_interval = np.linspace(1, -1, int(2 / self.topk_stride) + 1)

            ii = 0
            for h, l in zip(stride_interval[:-1], stride_interval[1:]):
                if len(out_ind) == topk:
                    break

                temp_ind, temp_val = [], []
                while sroted_oks[ii] >= l and sroted_oks[ii] < h:
                    temp_ind.append(topk_ind[ii])
                    temp_val.append(sroted_oks[ii])
                    ii += 1
                if self.topk_thr >= l and len(temp_ind) > 0:
                    ind = random.choice(temp_ind)
                    out_ind.append(int(ind))
                    out_val.append(oks[ind])
        else:
            while oks[topk_ind[0]] > self.topk_thr:
                topk_ind = np.delete(topk_ind, 0)
            out_ind = [int(topk_ind[i]) for i in range(topk)]
            out_val = [oks[i] for i in out_ind]

        return out_ind, out_val

    def topk_pose(self, pose, topk=1, pose_label="dts"):
        w, h = pose['width'], pose['height']

        # Normalize pose
        norm_pose, affine_params = self.normalize_pose(pose, self.crop_size)
        n_ratio, pad_shift, bbox = affine_params

        # 计算OKS相似度
        oks = compute_OKS(norm_pose, bbox[None], self.pose_lib["body"][pose_label])[0]
        # topk采样
        topk_ind, topk_oks = self.topk_sampling(oks, topk)

        out_pose_img = []
        for ind, pose_similarity in zip(topk_ind, topk_oks):
            body_pose = self.pose_lib["body"][pose_label][ind: ind + 1]
            if self.draw_hand:
                hand_pose = self.pose_lib["hand"][pose_label][ind]

            body_pose[..., :2] /= n_ratio
            body_pose[..., :2] -= np.array(pad_shift)
            body_pose[..., :2] += bbox[:2]
            if self.draw_hand:
                hand_pose[..., :2] /= n_ratio
                hand_pose[..., :2] -= np.array(pad_shift)
                hand_pose[..., :2] += bbox[:2]
            out_pose_img.append(
                self.draw_pose(body_pose, (h, w),
                               hand_pose=hand_pose if self.draw_hand else None))

        return out_pose_img, (topk_ind, topk_oks)

    def __call__(self, image, topk=1, pose_label="dts", **kwargs):
        # 推理pose
        src_pose_img, pose = self.pose_infer(image)
        if len(pose['body']['bboxes']) == 0:
            return None, None, None
        src_pose_img = Image.fromarray(src_pose_img)

        # topk相似pose
        topk_pose_img, topk_info = self.topk_pose(pose, topk, pose_label)

        generator = get_torch_generator(get_fixed_seed(-1), device=self.device)
        # ip-adapter image
        ip_adapter_image = self.crop_human_image(image, pose)
        ip_adapter_image, _ = self.pad_image(ip_adapter_image)

        out_img, out_pose = [], []
        gender = "person" if pose_label == "dts" else pose_label
        for pose_img in topk_pose_img:
            res_img = self.pipe_pose_trans(
                prompt=f"{gender} posing for a photo, high-res, best quality",
                negative_prompt="(worst quality:2), (low quality:2), nude, lowres",
                image=image,
                control_image=pose_img,
                ip_adapter_image=ip_adapter_image,
                height=self.infer_size[0],
                width=self.infer_size[1],
                strength=0.95,
                num_inference_steps=20,
                guidance_scale=3.5,
                num_images_per_prompt=1,
                generator=generator,
            )
            res_img = self.codeformer(res_img)[0]

            out_img.append(res_img)
            out_pose.append(pose_img)

        out_pose.insert(0, src_pose_img)
        return out_img, out_pose, topk_info


def get_img_path_list(img_file=None, img_dir=None):
    img_path_list = []
    if img_file is None and img_dir is None:
        raise Exception("Please specify `ref_img` or `ref_dir`.")
    elif img_file is not None and img_dir is None:
        # image
        assert exists(img_file)
        img_path_list.append(img_file)
    elif img_file is None and img_dir is not None:
        # dir
        assert exists(img_dir)
        img_path_list = os.listdir(img_dir)
        img_path_list = [join(img_dir, a) for a in img_path_list if splitext(
            a)[1].lower() in ['.jpg', '.jpeg', '.png']]
    else:
        raise Exception("`ref_img` and `ref_dir` cannot both be assigned.")

    return img_path_list


def read_image(args, file_path):
    if args.vos_pkl is None:
        img = load_image(file_path)
    else:
        img = args.vos_client.download_vos_pil(file_path)
        img = load_image(img)

    return img


def main(args):
    pose_trans = PoseTransfer(args)
    if args.vos_pkl is None:
        img_list = get_img_path_list(args.ref_img, args.ref_dir)
    else:
        img_list = load_file(args.vos_pkl)
        args.vos_client = VOSClient()

    if args.rank is not None and args.num_ranks is not None:
        assert args.rank < args.num_ranks
        total_num = len(img_list)
        stride = int(total_num / args.num_ranks)

        start_idx = stride * args.rank
        end_idx = stride * (args.rank + 1) if args.rank + 1 < args.num_ranks else len(img_list)
        # TODO: Clean here
        img_keys = sorted(img_list.keys())
        img_keys = img_keys[start_idx: end_idx]
        img_list = [img_list[k] for k in img_keys]

    for i, file in enumerate(img_list):
        try:
            t1 = time.time()
            print(f"{i + 1}/{len(img_list)}: {file}")
            name = basename(file)
            ref_img = read_image(args, file)
            crop_img, _, _ = pose_trans.crop_image(ref_img, args.resolution)
            if crop_img is None:
                print(f"Error: input image no bbox: {file}")
                continue

            pose_label = name.split('_')[0]
            pose_label = "dts" if pose_label not in gender_map else gender_map[pose_label]
            out_imgs, out_pose, topk_info = pose_trans(
                crop_img, topk=args.topk, pose_label=pose_label)
            if out_imgs is None:
                print(f"Error: input image no pose: {file}")
                continue

            if args.display:
                out_imgs = np.concatenate([crop_img, ] + out_imgs, axis=1)
                out_pose = np.concatenate(out_pose, axis=1)
                out_imgs = np.concatenate([out_imgs, out_pose])
                Image.fromarray(out_imgs).save(join(args.out_dir, splitext(name)[0] + '.jpg'))
            else:
                for j, (im, ind_, oks_) in enumerate(zip(out_imgs, topk_info[0], topk_info[1])):
                    # out_im = np.concatenate([crop_img, im], axis=1)
                    # Image.fromarray(out_im).save(
                    #     join(args.out_dir, splitext(name)[0] + f'_{ind_}_{round(oks_, 2)}_{j}.jpg'))
                    im.save(join(args.out_dir, splitext(name)[0] + f'_{j}.jpg'))
            t2 = time.time()
            print(f"time: {t2 - t1}s")
        except Exception as e:
            print(f"Unknown error: {e}")
            continue


if __name__ == '__main__':
    args = parse_args()
    main(args)
    print('Done!')
