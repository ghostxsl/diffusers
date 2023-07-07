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
from diffusers.pipelines.vip.vip_pose_transfer import VIPPoseTransferPipeline
from diffusers.schedulers import EulerAncestralDiscreteScheduler, KDPMPP2MDiscreteScheduler
from diffusers.models.controlnetxs import ControlNetXSModel
from diffusers.models.referencenet import ReferenceNetModel
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers.models.controlnet import ControlNetModel
from diffusers.pipelines.controlnet import VIPStableDiffusionControlNetInpaintPipeline
from diffusers.utils.prompt_parser import load_webui_textual_inversion

from diffusers.utils.vip_utils import *
from diffusers.data.utils import *
from diffusers.data.vos_client import VOSClient

from aistudio.utils.loader import ROOT_DIR
from aistudio.extensions.HumanPose import HumanPose
from aistudio.extensions.HumanPose.utils import POSE_CONFIG_DIR
from aistudio.extensions.P3MNet import P3MNet
from aistudio.extensions.LamaInpaint import LamaInpaint
from aistudio.extensions.CodeFormer import CodeFormer


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
        default=0.9,
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
        "--restore",
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
        "--restore_model_path",
        default="/apps/dat/cv/xsl/weights/majicMIX-realistic-7",
        type=str,
        help="Path to restore model.")

    parser.add_argument(
        "--resolution",
        type=str,
        default="768x576",
        help=(
            "The resolution(h x w) for input images, all the images"
            " will be resized to this resolution."
        ),
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="/apps/dat/cv/xsl/exp_animate/vip_230k",
    )
    parser.add_argument(
        "--referencenet_model_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--vae_model_path",
        type=str,
        default="/apps/dat/cv/xsl/weights/sd-image-variations/vae",
    )
    parser.add_argument(
        "--image_encoder_model_path",
        type=str,
        default="/apps/dat/cv/xsl/weights/sd-image-variations/image_encoder",
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
    def __init__(self, args):
        self.args = args
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

        vae = AutoencoderKL.from_pretrained(args.vae_model_path).to(self.device, dtype=self.dtype)
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            args.image_encoder_model_path).to(self.device, dtype=self.dtype)
        unet = UNet2DConditionModel.from_pretrained(
            args.base_model_path, subfolder="unet").to(self.device, dtype=self.dtype)
        controlnet = ControlNetXSModel.from_pretrained(
            args.base_model_path, subfolder="controlnet").to(self.device, dtype=self.dtype)
        if args.referencenet_model_path:
            referencenet = ReferenceNetModel.from_pretrained(
                args.referencenet_model_path).to(self.device, dtype=self.dtype)
        else:
            referencenet = ReferenceNetModel.from_pretrained(
                args.base_model_path, subfolder="referencenet").to(self.device, dtype=self.dtype)
        scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
            args.base_model_path, subfolder="scheduler")
        self.pipe_pose_trans = VIPPoseTransferPipeline(
            vae=vae,
            unet=unet,
            scheduler=scheduler,
            feature_extractor=CLIPImageProcessor(),
            image_encoder=image_encoder,
            controlnet=controlnet,
            referencenet=referencenet).to(self.device)

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
        self.matting_infer = P3MNet(
            model_path=join(self.weight_dir, "extensions/p3mnet_epoch50.pth"),
            infer_size=640,
            device=self.device,
        )
        self.lama_infer = LamaInpaint(
            train_config_path=join(self.weight_dir, "extensions/big-lama/config.yaml"),
            lama_config_path=join(self.weight_dir, "extensions/big-lama/default.yaml"),
            model_path=join(self.weight_dir, "extensions/big-lama"),
            device=self.device,
        )

        if args.restore:
            controlnet = ControlNetModel.from_pretrained(
                join(self.weight_dir, "controlnet/control_v11p_sd15_openpose"),
                torch_dtype=self.dtype).to(self.device)

            self.restore_pipe = VIPStableDiffusionControlNetInpaintPipeline.from_pretrained(
                args.restore_model_path,
                controlnet=controlnet, torch_dtype=self.dtype).to(self.device)

            load_webui_textual_inversion(join(self.weight_dir, "embedding"), self.restore_pipe)
            # self.restore_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.restore_pipe.scheduler.config)
            self.restore_pipe.scheduler = KDPMPP2MDiscreteScheduler.from_config(self.restore_pipe.scheduler.config)
            self.codeformer = CodeFormer(
                model_path=join(self.weight_dir, "extensions/codeformer-v0.1.0.pth"),
                det_model_path=join(self.weight_dir, "extensions/detection_Resnet50_Final.pth"),
                parsing_model_path=join(self.weight_dir, "extensions/parsing_parsenet.pth"),
                torch_compile=False,
                device=self.device,
            )

    def load_pose_lib(self, pose_dir):
        print("Load pose lib...")
        ch, cw = self.infer_size
        out = []
        for name in tqdm(os.listdir(pose_dir)):
            try:
                pose = load_file(join(pose_dir, name))
                w, h = pose['width'], pose['height']
                bboxes = pose['body']['bboxes'] * np.array([w, h, w, h])
                if len(bboxes) == 0:
                    continue
                # 取得分最高的bbox
                bbox = bboxes[0]

                body_kpts = pose['body']['keypoints'][0] * np.array([w, h, 1.])
                bbox[:2] -= 2; bbox[-2:] += 2
                body_kpts[..., :2] -= bbox[:2]

                # 按比例resize
                bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
                ratio_h, ratio_w = ch / bh, cw / bw
                if ratio_h < ratio_w:
                    # 按高 resize
                    ow = int(bh / ch * cw)
                    expand_w = ow - bw
                    body_kpts[..., 0] += int(expand_w / 2)
                    body_kpts[..., :2] *= ratio_h
                elif ratio_h > ratio_w:
                    # 按宽 resize
                    oh = int(bw / cw * ch)
                    expand_h = oh - bh
                    body_kpts[..., 1] += int(expand_h / 2)
                    body_kpts[..., :2] *= ratio_w
                else:
                    body_kpts[..., :2] *= ratio_h

                out.append(body_kpts)
            except:
                continue
        return {
            "dts": np.stack(out)
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

    def crop_image(self, image):
        image = np.array(image)
        bboxes, _ = self.pose_infer.detector(image)
        if len(bboxes) == 0:
            return None, None, None

        # 取得分最高的bbox
        det_bbox = np.int32(bboxes[0])
        crop_bbox, pad_ = crop_human_bbox(det_bbox, image.shape[:2], self.infer_size)
        x1, y1, x2, y2 = crop_bbox
        crop_img = image[y1: y2, x1: x2]
        if pad_ is not None:
            crop_img = np.pad(crop_img, pad_, constant_values=255)
        crop_img = Image.fromarray(crop_img).resize(self.infer_size[::-1], Image.LANCZOS)

        return crop_img, crop_bbox, pad_

    def matting_image(self, image):
        label_matting = self.matting_infer(image)
        if np.sum(label_matting) == 0:
            return None

        label_matting = label_matting[..., None].astype('float32') / 255.0
        ref_img = np.array(image).astype('float32')
        bg = np.ones_like(ref_img) * 255.0
        ref_img = ref_img * label_matting + bg * (1 - label_matting)
        ref_img = Image.fromarray(np.clip(ref_img, 0, 255).astype('uint8'))

        return ref_img

    def compound_image(self, gen_imgs, gen_imgs_pose, ref_img, crop_bbox, pad_border):
        # 获取lama背景图
        ref_mask = self.matting_infer(ref_img)
        if np.sum(ref_mask) == 0:
            return None
        lama_img, _, _ = self.lama_infer(ref_img, ref_mask)

        cw, ch = crop_bbox[2] - crop_bbox[0], crop_bbox[3] - crop_bbox[1]

        def affine_crop(img, cw, ch):
            # 仿射变换回到原始crop尺寸
            if pad_border is None or (sum(pad_border[0]) == 0 and sum(pad_border[1]) == 0):
                pass
            else:
                if sum(pad_border[0]) != 0:
                    img = img.resize((cw, ch + sum(pad_border[0])), 1)
                    y1 = pad_border[0][0]
                    img = np.array(img)[y1: ch + y1]
                elif sum(pad_border[1]) != 0:
                    img = img.resize((cw + sum(pad_border[1]), ch), 1)
                    x1 = pad_border[1][0]
                    img = np.array(img)[:, x1: cw + x1]
                else:
                    raise Exception('error pad_border!')
                img = Image.fromarray(img)

            return img.resize((cw, ch), 1)

        # 将生成图放回原图位置
        out_imgs = [affine_crop(im, cw, ch) for im in gen_imgs]
        for i, res_img in enumerate(out_imgs):
            label_matting = self.matting_infer(res_img)

            label_matting = label_matting[..., None].astype('float32') / 255.0
            res_img = np.array(res_img).astype('float32')
            # pose迁移后的前景放回原图位置
            temp_img = lama_img.copy()
            crop_bg = temp_img[crop_bbox[1]: crop_bbox[3], crop_bbox[0]: crop_bbox[2]]
            crop_bg = res_img * label_matting + crop_bg * (1 - label_matting)
            crop_bg = np.clip(crop_bg, 0, 255).astype('uint8')
            temp_img[crop_bbox[1]: crop_bbox[3], crop_bbox[0]: crop_bbox[2]] = crop_bg

            out_imgs[i] = Image.fromarray(temp_img)

        # 将pose图放回原图位置
        ref_pose = Image.new("RGB", ref_img.size, 0)
        out_pose = [affine_crop(im, cw, ch) for im in gen_imgs_pose]
        for i, pose_img in enumerate(out_pose):
            temp_pose = ref_pose.copy()
            temp_pose = np.array(temp_pose)
            temp_pose[crop_bbox[1]: crop_bbox[3], crop_bbox[0]: crop_bbox[2]] = np.array(pose_img)

            out_pose[i] = Image.fromarray(temp_pose)

        return out_imgs, out_pose

    def restore_image(self,
                      images,
                      pose_images,
                      gender="person",
                      img_size=(1024, 1024)):

        def remove_border(restore_img, pad_border=None):
            if pad_border is not None:
                restore_img = np.array(restore_img)
                if sum(pad_border[0]) != 0:
                    y1 = pad_border[0][0]
                    y2 = -pad_border[0][1] if pad_border[0][1] != 0 else restore_img.shape[0]
                    restore_img = restore_img[y1:y2]
                elif sum(pad_border[1]) != 0:
                    x1 = pad_border[1][0]
                    x2 = -pad_border[1][1] if pad_border[1][1] != 0 else restore_img.shape[1]
                    restore_img = restore_img[:, x1:x2]
                restore_img = Image.fromarray(restore_img)
            return restore_img

        out_imgs = []
        mask_image = Image.new("RGB", img_size, (255, 255, 255))
        for img, pose_img in zip(images, pose_images):
            pad_img, pad_border = self.pad_image(img)
            pad_pose_img, _ = self.pad_image(pose_img, pad_values=0)
            seed = get_fixed_seed(-1)
            generator = get_torch_generator(seed, device=self.device)
            inter_imgs, image_overlay = self.restore_pipe(
                image=pad_img,
                mask_image=mask_image,
                prompt=f"{gender} posing for a photo, 4k, high-res, masterpiece, best quality, sharp focus, (cinematic lighting), soft lighting, dynamic angle",
                negative_prompt="ng_deepnegative_v1_75t, (nsfw:2), (naked:2), (greyscale:1.2), paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), acnes, age spot, glans",
                control_image=pad_pose_img,
                height=img_size[0],
                width=img_size[1],
                strength=0.4,
                num_inference_steps=30,
                guidance_scale=2.5,
                num_images_per_prompt=1,
                generator=generator,
                controlnet_conditioning_scale=1.0,
            )
            inter_imgs = alpha_composite(inter_imgs, image_overlay)
            restore_img = self.codeformer(inter_imgs)[0]

            restore_img = remove_border(restore_img.resize(pad_img.size, 1), pad_border)
            out_imgs.append(restore_img.resize(img.size, 1))
        return out_imgs

    def draw_pose(self, pose, draw_size, kpt_thr=0.3):
        height, width = draw_size
        canvas = np.zeros(shape=(height, width, 3), dtype=np.uint8)
        canvas = draw_bodypose(canvas, pose[..., :2], pose[..., 2] > kpt_thr)
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
        bboxes = pose['body']['bboxes']
        kpts = pose['body']['keypoints']
        # 取得分最高的bbox
        bbox = bboxes[0: 1] * np.array([w, h, w, h])
        kpts = kpts[0: 1] * np.array([w, h, 1.])
        # 计算OKS相似度
        src_pose_img = self.draw_pose(kpts, self.infer_size)
        oks = compute_OKS(kpts, bbox, self.pose_lib[pose_label])[0]

        topk_ind, topk_oks = self.topk_sampling(oks, topk)

        out_pose_img = []
        for ind, pose_similarity in zip(topk_ind, topk_oks):
            pose = self.pose_lib[pose_label][ind]
            out_pose_img.append(self.draw_pose(pose[None], self.infer_size))

        return out_pose_img, src_pose_img, (topk_ind, topk_oks)

    def __call__(self, image, topk=1, pose_label="dts", **kwargs):
        # 推理pose
        _, pose = self.pose_infer(image, body=False, hand=False)
        if len(pose['body']['bboxes']) == 0:
            return None, None

        # topk相似pose
        topk_pose_img, src_pose_img, topk_info = self.topk_pose(pose, topk, pose_label)

        out_img = []
        generator = get_torch_generator(get_fixed_seed(-1), device=self.device)
        for pose_img in topk_pose_img:
            res_img = self.pipe_pose_trans(
                reference_image=image,
                control_image=pose_img,
                height=self.infer_size[0],
                width=self.infer_size[1],
                num_inference_steps=25,
                guidance_scale=2.0,
                num_images_per_prompt=1,
                generator=generator,
            )[0]
            out_img.append(res_img)
        topk_pose_img.insert(0, src_pose_img)
        return out_img, topk_pose_img, topk_info


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
            # TODO: Clean here
            if args.vos_pkl is not None:
                race = file.get("人种", "亚洲")
                race = race_map.get(race, "asian")
                file = file["vos路径"]

            print(f"{i + 1}: {file}")
            name = basename(file)
            ref_img = read_image(args, file)
            crop_img, crop_bbox, pad_border = pose_trans.crop_image(ref_img)
            if crop_img is None:
                print(f"Error1: input image no pose: {file}")
                continue

            matting_img = pose_trans.matting_image(crop_img)
            if matting_img is None:
                print(f"Error2: crop image no matting: {file}")
                continue

            pose_label = name.split('_')[0]
            pose_label = "dts" if pose_label not in gender_map else gender_map[pose_label]
            res, dis_pose, topk_info = pose_trans(
                matting_img, topk=args.topk, pose_label=pose_label)
            if res is None:
                print(f"Error3: matting image no pose: {file}")
                continue

            out_imgs, dis_pose = pose_trans.compound_image(
                res, dis_pose, ref_img, crop_bbox, pad_border)
            if out_imgs is None:
                print(f"Error4: compound image no matting: {file}")
                out_imgs = [a.resize(ref_img.size, 1) for a in res]

            if args.restore:
                gender = pose_label if pose_label in gender_map else "person"
                if args.vos_pkl is not None:
                    gender = race + " " + gender
                restore_imgs = pose_trans.restore_image(out_imgs, dis_pose[1:], gender)

            if args.display:
                out_imgs = np.concatenate([ref_img, ] + out_imgs, axis=1)
                dis_pose = [a.resize(ref_img.size, 1) for a in dis_pose]
                dis_pose = np.concatenate(dis_pose, axis=1)
                if args.restore:
                    restore_imgs = np.concatenate([ref_img, ] + restore_imgs, axis=1)
                    out_imgs = np.concatenate([out_imgs, restore_imgs, dis_pose])
                else:
                    out_imgs = np.concatenate([out_imgs, dis_pose])
                Image.fromarray(out_imgs).save(join(args.out_dir, splitext(name)[0] + '.jpg'))
            else:
                if args.restore:
                    out_imgs = restore_imgs
                for j, (im, ind_, oks_) in enumerate(zip(out_imgs, topk_info[0], topk_info[1])):
                    # out_im = np.concatenate([ref_img, im], axis=1)
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
