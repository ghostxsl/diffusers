import argparse
import os
from os.path import splitext, join

from tqdm import tqdm
import csv
import numpy as np
from PIL import Image

from diffusers.utils.vip_utils import load_image
from diffusers.data.face_train import FaceProcess, resize_img


IMAGE_EXTS = ['.jpg', '.jpeg', '.png']


def parse_args():
    parser = argparse.ArgumentParser(description="Data processing script for training face lora model.")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Directory to face images.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output_face",
        help="Directory to save processed face images.",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=" ",
        help="Text prompts for training.",
    )
    parser.add_argument(
        "--yolov8_path",
        type=str,
        default="yolov8n-face.onnx",
    )
    parser.add_argument(
        "--human_parsing_path",
        type=str,
        default="cv_resnet101_image-multiple-human-parsing",
    )
    parser.add_argument(
        "--face_landmark_path",
        type=str,
        default="cv_manual_facial-landmark-confidence_flcm",
    )
    parser.add_argument(
        "--face_rotate",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--size",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--face_thr",
        type=float,
        default=0.855,
    )

    args = parser.parse_args()
    return args


def main(args):
    face_process = FaceProcess(
        yolov8_path=args.yolov8_path,
        human_parsing_path=args.human_parsing_path,
        face_landmark_path=args.face_landmark_path,
        face_rotate=args.face_rotate,
        size=args.size,
        face_thr=args.face_thr,
    )
    img_dir = args.image
    img_list = os.listdir(img_dir)
    img_list = [a for a in img_list if splitext(a)[-1].lower() in IMAGE_EXTS]

    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)

    f = open(join(out_dir, "metadata.csv"), "a", encoding="utf-8", newline="")
    writer = csv.writer(f)
    writer.writerow(['file_name', 'text'])

    idx = 0
    for name in tqdm(img_list):
        img = load_image(join(img_dir, name))
        out_img = face_process(img)
        if out_img is not None:
            save_path = join(out_dir, f"{idx}.png")
            out_img.save(save_path)
            writer.writerow([f"{idx}.png", args.text])
            f.flush()
            print(f"[{name}]: {save_path} saved.")
            idx += 1
        else:
            print(f"{name} processing failed.")

    f.close()
    print("Done!")


if __name__ == '__main__':
    args = parse_args()
    main(args)
