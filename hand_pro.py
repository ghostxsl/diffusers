import pandas as pd
from PIL import Image
from os.path import join, splitext
import os
import shutil
import hashlib
import csv
import cv2
import numpy as np
import mmcv
from extensions.HumanParsing.inference.inference_single import HumanParser

from diffusers.utils.vip_utils import load_image, mediapipe_face_detection, get_crop_region, expand_crop_region

human_parsing = HumanParser(
    model_path="/xsl/wilson.xu/weights/deeplabv3plus-xception-vocNov14_20-51-38_epoch-89.pth")


img_dir = "/xsl/wilson.xu/dataset/face_images/#10_progress1_as_female1_1024x1024"
img_list = os.listdir(img_dir)

save_dir = f'./output'
os.makedirs(save_dir, exist_ok=True)

out_file = "/xsl/wilson.xu/dataset/face_images/#10_progress1_as_female1_1024x1024.csv"
f = open(out_file, "a", encoding="utf-8", newline="")
writer = csv.writer(f)
writer.writerow(['file_name', 'text'])

for i, name in enumerate(img_list):
    if splitext(name)[-1].lower() in ['.jpg', '.jpeg', '.png'] and '_parsing' not in name:
        img = load_image(join(img_dir, name))
        # img.save(join(img_dir, name))
        bboxes, masks, preview = mediapipe_face_detection(img)
        crop_region = get_crop_region(np.array(masks[0]), pad=15)

        label_parsing = human_parsing.inference(img).astype('uint8')
        Image.fromarray(label_parsing).save(join(img_dir, splitext(name)[0] + '_parsing.jpg'))

        writer.writerow([name, " "])
        f.flush()

        mask = np.zeros_like(label_parsing)
        for idx in [1, 2, 13]:
            mask[label_parsing == idx] = 255
        kernel = np.ones([5, 5], np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        x = np.where(np.sum(mask, axis=0))[0]
        x1, x2 = x[0], x[-1]
        y = np.where(np.sum(mask, axis=1))[0]
        y1, y2 = y[0], y[-1]

        canvas = mmcv.imshow_bboxes(np.array(img), np.array([[x1, y1, x2, y2]]), 'green', show=False)
        canvas = mmcv.imshow_bboxes(canvas, np.array([crop_region]), 'blue', show=False)
        Image.fromarray(canvas).save(join(save_dir, name))

        print(i, name)


f.close()
print("Done")

# import os

# from PIL import Image
# import numpy as np
# import pandas as pd

# from mmengine import mkdir_or_exist
# from mmpose.blendpose.inferencer import VIPPoseInferencer
# from mmpose.blendpose.utils import pkl_save


# def main():
#     csv_list = pd.read_csv("/xsl/wilson.xu/dataset/49w_with_hand_caption.csv").values.tolist()
#     img_dir = "/xsl/wilson.xu/dataset/train_images"
#     out_dir = "/xsl/wilson.xu/dataset/controlnet_pose"
#     mkdir_or_exist(out_dir)
#     mkdir_or_exist(os.path.join(out_dir, "pose"))
#     mkdir_or_exist(os.path.join(out_dir, "res"))

#     root_dir = "/xsl/wilson.xu"

#     det_config = f"{root_dir}/mmpose/mmpose/blendpose/configs/rtmdet_l_8xb32-300e_coco.py"
#     det_checkpoint = f"{root_dir}/weights/rtmpose_model/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth"
#     body_cfg = f"{root_dir}/mmpose/mmpose/blendpose/configs/rtmpose-l_8xb256-420e_body8-256x192.py"
#     body_pth = f"{root_dir}/weights/rtmpose_model/rtmpose-l_simcc-body7_pt-body7_420e-256x192-4dba18fc_20230504.pth"
#     wholebody_cfg = f"{root_dir}/mmpose/mmpose/blendpose/configs/dwpose_l_wholebody_384x288.py"
#     wholebody_pth = f"{root_dir}/weights/rtmpose_model/dw-ll_ucoco_384.pth"
#     device = "cuda"

#     inferencer = VIPPoseInferencer(
#         det_config, det_checkpoint,
#         bodypose_cfg=body_cfg,
#         bodypose_pth=body_pth,
#         wholebodypose_cfg=wholebody_cfg,
#         wholebodypose_pth=wholebody_pth,
#         return_results=True,
#         body_kpt_thr=0.3,
#         hand_kpt_thr=0.3,
#         device=device)

#     for i, (name, caption) in enumerate(csv_list):
#         if os.path.splitext(name)[-1].lower() in ['.jpg', '.jpeg', '.png']:
#             # inference
#             img_file = os.path.join(img_dir, name)
#             img = np.asarray(Image.open(img_file).convert("RGB"))
#             canvas, results = inferencer(img.copy())
#             pkl_save(results,
#                      os.path.join(out_dir, "pose", os.path.splitext(name)[0] + "_pose.pkl"))
#             save_name = os.path.join(out_dir, "res", name)
#             Image.fromarray(canvas).save(save_name)
#             print(f"{i + 1}/{len(csv_list)}: {save_name} has been saved")

# if __name__ == '__main__':
#     main()
#     print("Done!")
