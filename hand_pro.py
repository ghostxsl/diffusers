import pandas as pd
from PIL import Image
from os.path import join, splitext
import os
import shutil
import hashlib
import csv
import cv2
import numpy as np


img_dir = "/Users/wilson.xu/Downloads/new_hands"
out_dir = "/Users/wilson.xu/Downloads/out_hands"
out_list = [splitext(a)[0] for a in os.listdir(out_dir)]

img_list = os.listdir(img_dir)
for i, name in enumerate(img_list):
    src = join(img_dir, name)
    with open(src, 'rb') as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    if file_hash not in out_list:
        dst = join(out_dir, file_hash + splitext(name)[-1])
        shutil.copy(src, dst)
        print(i, name)
print("Done!")


import os

from PIL import Image
import numpy as np
import pandas as pd

from mmengine import mkdir_or_exist
from mmpose.blendpose.inferencer import VIPPoseInferencer
from mmpose.blendpose.utils import pkl_save


def main():
    csv_list = pd.read_csv("/xsl/wilson.xu/dataset/49w_with_hand_caption.csv").values.tolist()
    img_dir = "/xsl/wilson.xu/dataset/train_images"
    out_dir = "/xsl/wilson.xu/dataset/controlnet_pose"
    mkdir_or_exist(out_dir)
    mkdir_or_exist(os.path.join(out_dir, "pose"))
    mkdir_or_exist(os.path.join(out_dir, "res"))

    root_dir = "/xsl/wilson.xu"

    det_config = f"{root_dir}/mmpose/mmpose/blendpose/configs/rtmdet_l_8xb32-300e_coco.py"
    det_checkpoint = f"{root_dir}/weights/rtmpose_model/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth"
    body_cfg = f"{root_dir}/mmpose/mmpose/blendpose/configs/rtmpose-l_8xb256-420e_body8-256x192.py"
    body_pth = f"{root_dir}/weights/rtmpose_model/rtmpose-l_simcc-body7_pt-body7_420e-256x192-4dba18fc_20230504.pth"
    wholebody_cfg = f"{root_dir}/mmpose/mmpose/blendpose/configs/dwpose_l_wholebody_384x288.py"
    wholebody_pth = f"{root_dir}/weights/rtmpose_model/dw-ll_ucoco_384.pth"
    device = "cuda"

    inferencer = VIPPoseInferencer(
        det_config, det_checkpoint,
        bodypose_cfg=body_cfg,
        bodypose_pth=body_pth,
        wholebodypose_cfg=wholebody_cfg,
        wholebodypose_pth=wholebody_pth,
        return_results=True,
        body_kpt_thr=0.3,
        hand_kpt_thr=0.3,
        device=device)

    for i, (name, caption) in enumerate(csv_list):
        if os.path.splitext(name)[-1].lower() in ['.jpg', '.jpeg', '.png']:
            # inference
            img_file = os.path.join(img_dir, name)
            img = np.asarray(Image.open(img_file).convert("RGB"))
            canvas, results = inferencer(img.copy())
            pkl_save(results,
                     os.path.join(out_dir, "pose", os.path.splitext(name)[0] + "_pose.pkl"))
            save_name = os.path.join(out_dir, "res", name)
            Image.fromarray(canvas).save(save_name)
            print(f"{i + 1}/{len(csv_list)}: {save_name} has been saved")

if __name__ == '__main__':
    main()
    print("Done!")
