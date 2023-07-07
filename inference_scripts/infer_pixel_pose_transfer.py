# Copyright (c) wilson.xu. All rights reserved.
import os
import argparse
from os.path import join, splitext, exists, split
import numpy as np
from PIL import Image
import torch

from diffusers.pipelines.vip import VIPPixelPoseTransferPipeline
from diffusers.schedulers import EulerAncestralDiscreteScheduler, KDPMPP2MDiscreteScheduler
from diffusers.models.vip import ReferenceTextureModel, VIPUNet2DConditionModel
from diffusers.utils.vip_utils import *

from aistudio.utils.loader import ROOT_DIR
from aistudio.extensions.HumanPose import HumanPose
from aistudio.extensions.HumanPose.utils import POSE_CONFIG_DIR
from aistudio.extensions.P3MNet import P3MNet


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
        "--pose_img",
        default=None,
        type=str,
        help="Path to pose image.")
    parser.add_argument(
        "--pose_dir",
        default=None,
        type=str,
        help="Directory to pose image.")
    parser.add_argument(
        "--out_dir",
        default="output",
        type=str,
        help="Directory to save.")

    parser.add_argument(
        "--use_pose_infer",
        default=False,
        action="store_true",
        help="")
    parser.add_argument(
        "--crop",
        default=False,
        action="store_true",
        help="")
    parser.add_argument(
        "--matting",
        default=False,
        action="store_true",
        help="")
    parser.add_argument(
        "--use_pad",
        default=False,
        action="store_true",
        help="")

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
        default="/apps/dat/cv/xsl/exp_animate/checkpoint-130000",
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
        args.resolution = [int(args.resolution),] * 2
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


def get_crop_bbox(det_bbox, crop_size, img_size, pad_bbox=5):
    h, w = img_size
    x1, y1, x2, y2 = det_bbox
    x1 = 0 if x1 - pad_bbox < 0 else x1 - pad_bbox
    y1 = 0 if y1 - pad_bbox < 0 else y1 - pad_bbox
    x2 = w if x2 + pad_bbox > w else x2 + pad_bbox
    y2 = h if y2 + pad_bbox > h else y2 + pad_bbox

    bh, bw = y2 - y1, x2 - x1
    ch, cw = crop_size
    ratio_h, ratio_w = ch / bh, cw / bw

    pad_ = [[0, 0], [0, 0], [0, 0]]
    if ratio_h < ratio_w:
        # 按高 resize
        ow = int(bh / ch * cw)
        expand_w = ow - bw

        x1 -= int(expand_w / 2)
        if x1 < 0:
            pad_[1][0] = abs(x1)
            x1 = 0

        x2 += (expand_w - int(expand_w / 2))
        if x2 > w:
            pad_[1][1] = x2 - w
            x2 = w

        return [x1, y1, x2, y2], pad_
    elif ratio_h > ratio_w:
        # 按宽 resize
        oh = int(bw / cw * ch)
        expand_h = oh - bh

        y1 -= int(expand_h / 2)
        if y1 < 0:
            pad_[0][0] = abs(y1)
            y1 = 0

        y2 += (expand_h - int(expand_h / 2))
        if y2 > h:
            pad_[0][1] = y2 - h
            y2 = h

        return [x1, y1, x2, y2], pad_
    else:
        return det_bbox, None


def crop_image(pose_infer, image, crop_size=(1024, 768)):
    image = np.array(image)
    bboxes, _ = pose_infer.detector(image)

    if len(bboxes) == 0:
        return None
    elif len(bboxes) == 1:
        det_bbox = np.int32(bboxes[0])
    elif len(bboxes) > 1:
        area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        ind = np.argmax(area)
        det_bbox = np.int32(bboxes[ind])

    crop_bbox, pad_ = get_crop_bbox(det_bbox, crop_size, image.shape[:2])
    x1, y1, x2, y2 = crop_bbox
    crop_img = image[y1: y2, x1: x2]
    if pad_ is not None:
        crop_img = np.pad(crop_img, pad_, constant_values=255)

    return Image.fromarray(crop_img).resize(crop_size[::-1], Image.LANCZOS)


def pad_image(img, pad_values=255):
    w, h = img.size
    img = np.array(img)
    if w > h:
        pad_ = w - h
        img = np.pad(
            img,
            ((pad_ // 2, pad_ - pad_ // 2), (0, 0), (0, 0)),
            constant_values=pad_values
        )
    elif h > w:
        pad_ = h - w
        img = np.pad(
            img,
            ((0, 0), (pad_ // 2, pad_ - pad_ // 2), (0, 0)),
            constant_values=pad_values
        )
    return Image.fromarray(img)


def main(args):
    ref_img_list = get_img_path_list(args.ref_img, args.ref_dir)
    pose_img_list = get_img_path_list(args.pose_img, args.pose_dir)

    device = args.device
    dtype = args.dtype

    unet = VIPUNet2DConditionModel.from_pretrained(args.base_model_path, subfolder="unet").to(device, dtype=dtype)
    referencenet = ReferenceTextureModel.from_pretrained(args.base_model_path, subfolder="referencenet").to(device, dtype=dtype)
    scheduler = EulerAncestralDiscreteScheduler.from_pretrained(args.base_model_path, subfolder="scheduler")
    pipe = VIPPixelPoseTransferPipeline(
        unet=unet,
        scheduler=scheduler,
        referencenet=referencenet).to(device)

    weight_dir = args.weight_dir or join(ROOT_DIR, "weights")
    pose_infer = HumanPose(
        det_cfg=join(POSE_CONFIG_DIR, "rtmdet_l_8xb32-300e_coco.py"),
        det_pth=join(weight_dir, "extensions/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth"),
        bodypose_cfg=join(POSE_CONFIG_DIR, "rtmpose-l_8xb256-420e_body8-256x192.py"),
        bodypose_pth=join(weight_dir, "extensions/rtmpose-l_simcc-body7_pt-body7_420e-256x192-4dba18fc_20230504.pth"),
        wholebodypose_cfg=join(POSE_CONFIG_DIR, "dwpose_l_wholebody_384x288.py"),
        wholebodypose_pth=join(weight_dir, "extensions/dw-ll_ucoco_384.pth"),
        device=device
    )
    matting_infer = P3MNet(
        model_path=join(weight_dir, "extensions/p3mnet_epoch50.pth"),
        infer_size=640,
    )

    pose_list = []
    for file in pose_img_list:
        img = load_image(file)
        if args.use_pose_infer:
            if args.crop:
                img = crop_image(pose_infer, img)
            res, _ = pose_infer(img)
            img = Image.fromarray(res)
        if args.use_pad:
            img = pad_image(img, 0)
        pose_list.append(img)

    for i, file in enumerate(ref_img_list):
        ref_img = load_image(file)
        if args.crop:
            ref_img = crop_image(pose_infer, ref_img)
        if args.matting:
            label_matting = matting_infer(ref_img)
            label_matting = label_matting[..., None].astype('float32') / 255.0
            ref_img = np.array(ref_img).astype('float32')
            bg = np.ones_like(ref_img) * 255.0
            ref_img = ref_img * label_matting + bg * (1 - label_matting)
            ref_img = Image.fromarray(np.clip(ref_img, 0, 255).astype('uint8'))

        if args.use_pad:
            ref_img = pad_image(ref_img)
        print(f"{i + 1}: {file}")
        seed = get_fixed_seed(-1)
        generator = get_torch_generator(seed, device=device)

        for j, pose in enumerate(pose_list):
            out = pipe(
                reference_image=ref_img,
                control_image=pose,
                height=args.resolution[0],
                width=args.resolution[1],
                num_inference_steps=25,
                guidance_scale=2.0,
                num_images_per_prompt=1,
                generator=generator,
            )[0]
            out_img = np.concatenate(
                [ref_img.resize(args.resolution[::-1], 1),
                 pose.resize(args.resolution[::-1], 1),
                 out], axis=1)
            Image.fromarray(out_img).save(join(args.out_dir, splitext(split(file)[1])[0] + f"_{j}.jpg"))

if __name__ == '__main__':
    args = parse_args()
    main(args)
    print('Done!')
