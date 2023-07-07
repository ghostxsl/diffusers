import os
from os.path import join, splitext
import time
import requests, base64
import numpy as np
from io import BytesIO
import random
import torch
from scipy.interpolate import interp1d

import cv2
from PIL import Image, ImageEnhance, ImageFilter
from diffusers.models import ControlNetModel
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from diffusers.pipelines.controlnet.vip_pipeline_controlnet_inpaint import VIPStableDiffusionControlNetInpaintPipeline
from diffusers.schedulers import EulerAncestralDiscreteScheduler, KDPMPP2MDiscreteScheduler

from diffusers.utils.prompt_parser import load_webui_textual_inversion
from diffusers.utils.vip_utils import *

from mmpose.blendpose.inferencer import VIPPoseInferencer
from mmpose.blendpose.utils import pkl_load, ClothSeg
from extensions.CodeFormer.inference_codeformer_re import apply_codeformer
from extensions.HumanParsing.inference.inference_single import HumanParser
from extensions.midas import MidasDetector
from extensions.PIH import Evaluater


def postApi_zuotang(image):
    def pil2byte(pil):
        img_bytes = BytesIO()
        pil.save(img_bytes, format='JPEG')
        img_bytes = img_bytes.getvalue()
        return img_bytes

    if isinstance(image, str):
        input_image = open(image, 'rb')
    elif isinstance(image, Image.Image):
        input_image = pil2byte(image)
    else:
        raise ValueError("Incorrect format. Should be path or pil image.")
    mask = None
    try:
        response = requests.request(
            "POST",
            "https://techsz.aoscdn.com/api/tasks/visual/segmentation",
            headers={'X-API-KEY': 'wxpl80244enlgrv53'},
            data={'sync': '1', 'return_type': 2, 'output_type': 3},
            files={'image_file': input_image},
            verify=False
        )
        resp_json = response.json()
        print(resp_json['status'], resp_json['data']['progress'])
        if resp_json['status'] == 200:
            if resp_json['data']['progress'] == 100:
                mask_base64 = resp_json['data']['mask']
                # mask=base64ToimgArr(mask_base64)
                image_data = base64.b64decode(mask_base64)
                mask = Image.open(BytesIO(image_data))
                return mask
    except Exception as e:
        print(e)


# set init args
device = torch.device("cuda")
dtype = torch.float16
input_size = (1024, 1024)

# set init condition models
pose_path = f"/xsl/wilson.xu/weights/control_v11p_sd15_openpose"
canny_path = f"/xsl/wilson.xu/weights/control_v11p_sd15_canny"
depth_path = f"/xsl/wilson.xu/weights/control_sd15_depth"
pose_model = ControlNetModel.from_pretrained(pose_path, torch_dtype=dtype).to(device)
canny_model = ControlNetModel.from_pretrained(canny_path, torch_dtype=dtype).to(device)
depth_model = ControlNetModel.from_pretrained(depth_path, torch_dtype=dtype).to(device)

# set other models
# https://github.com/ghostxsl/mmpose
det_config = "/xsl/wilson.xu/mmpose/mmpose/blendpose/configs/rtmdet_l_8xb32-300e_coco.py"
det_checkpoint = "/xsl/wilson.xu/weights/rtmpose_model/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth"
body_config = "/xsl/wilson.xu/mmpose/configs/body_2d_keypoint/rtmpose/body8/rtmpose-l_8xb256-420e_body8-256x192.py"
body_checkpoint = "/xsl/wilson.xu/weights/rtmpose_model/rtmpose-l_simcc-body7_pt-body7_420e-256x192-4dba18fc_20230504.pth"
wholebody_cfg = "/xsl/wilson.xu/mmpose/mmpose/blendpose/configs/dwpose_l_wholebody_384x288.py"
wholebody_pth = "/xsl/wilson.xu/weights/rtmpose_model/dw-ll_ucoco_384.pth"
pose_inferencer = VIPPoseInferencer(det_config, det_checkpoint,
                                    body_config, body_checkpoint,
                                    wholebodypose_cfg=wholebody_cfg,
                                    wholebodypose_pth=wholebody_pth,
                                    body_kpt_thr=0.3,
                                    hand_kpt_thr=0.3,
                                    device=device)
face_restored_path = "/xsl/wilson.xu/weights/codeformer-v0.1.0.pth"

pih_path = "/xsl/wilson.xu/weights/ckpt_g39.pth"
pih_inferencer = Evaluater(pih_path, device=device)

clotheseg_path = "/xsl/wilson.xu/diffusers/extensions/ClotheSeg/cloth_seg_v5.0.pt"
clotheseg_inferencer = ClothSeg(clotheseg_path, device=device)

# build human parse model
human_parsing = HumanParser(
    model_path="/xsl/wilson.xu/weights/deeplabv3plus-xception-vocNov14_20-51-38_epoch-89.pth")
    # device=device)


def img_light_process(img):
    img = img.convert('L')
    # img = ImageOps.equalize(img, mask=None)
    img = img.resize((1536, 1536), resample=Image.BILINEAR)
    enhancer = ImageEnhance.Contrast(img)
    img_contrast = enhancer.enhance(1.3)
    img = img_contrast

    # 使用高斯模糊去除一部分细节
    img = img.filter(ImageFilter.GaussianBlur(radius=5))
    img = img.filter(ImageFilter.BoxBlur(radius=15))
    img = img.filter(ImageFilter.GaussianBlur(radius=10))

    # 使用中值滤波器去除更多的细节
    img = img.filter(ImageFilter.MedianFilter(size=3))
    img = img.filter(ImageFilter.SMOOTH)
    # 使用边缘增强滤波器保留光影和色块的模糊关系
    img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=1))

    gray = np.array(img)

    # 对图像进行自适应直方图均衡化
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    cl_img = clahe.apply(gray)

    # 将图像转换为 Pillow 格式，并显示处理后的图像
    img = Image.fromarray(cl_img)
    img = img.resize((img.size[0] // 16, img.size[1] // 16), resample=Image.BOX)
    img = img.resize((img.size[0] * 16, img.size[1] * 16), resample=Image.NEAREST)
    img = img.filter(ImageFilter.GaussianBlur(radius=10))
    enhancer = ImageEnhance.Contrast(img)
    img_contrast = enhancer.enhance(1.3)
    img = img_contrast
    img = img.resize(input_size, resample=Image.BILINEAR)

    return img.convert("RGB")


class Preprocesser():
    def __init__(self):
        pass

    def pad_img(self, img_file, is_upper=False):
        init_image = load_image(img_file)
        if init_image.size[0] < init_image.size[1]:
            bgImg: Image.Image = Image.new("RGB", (init_image.size[1], init_image.size[1]), (255, 255, 255))
            bgImg.paste(init_image, (round((init_image.size[1] - init_image.size[0]) / 2), 0))
            init_image = bgImg.resize(input_size)
        else:
            bgImg: Image.Image = Image.new("RGB", (init_image.size[0], init_image.size[0]), (255, 255, 255))
            if is_upper:
                bgImg.paste(init_image, (0, round(init_image.size[0] - init_image.size[1])))
            else:
                bgImg.paste(init_image, (0, round((init_image.size[0] - init_image.size[1]) / 2)))
            init_image = bgImg.resize(input_size)
        return init_image

    def check_body(self, kpts):
        upper, lower = 1, 1  # upper=检测到上半身人体关键点, lower=检测到下半身关键点, 0=检测到，1=没检测到
        kps_mote_valid = [kpt[-1] > 0.3 for kpt in kpts["body"]["keypoints"][0]]
        if kps_mote_valid[0] and kps_mote_valid[1] and (kps_mote_valid[2] or kps_mote_valid[5]) and (
                kps_mote_valid[14] or kps_mote_valid[15]):
            upper = 0
            if kps_mote_valid[10] or kps_mote_valid[13]:
                lower = 0
        return upper, lower

    def __call__(self, mote, bg_lst, is_seedfolder):
        pose_mote, kpts_mote = pose_inferencer(np.array(mote.convert("RGB")),
                                               return_results=True,
                                               hand=False)
        upper_mote, lower_mote = self.check_body(kpts_mote)
        if lower_mote:
            mote = self.pad_img(mote, is_upper=True)
        else:
            mote = self.pad_img(mote, is_upper=False)

        if len(bg_lst) == 0:  # 换模特任务
            # 0=上传模特图符合要求, 1=模特图中未检测到上半身关键点
            if upper_mote:
                return [1]
            return [0, mote]

        kpts = list()
        if not is_seedfolder:  # 背景图是用户上传图
            bg = bg_lst[0]
            pose_bg, kpts_bg = pose_inferencer(np.array(bg.convert("RGB")),
                                               return_results=True,
                                               hand=False)
            upper_bg, lower_bg = self.check_body(kpts_bg)
            if lower_bg:
                bg = self.pad_img(bg, is_upper=True)
            else:
                bg = self.pad_img(bg, is_upper=False)
            kpts = [kpts_mote["body"]["keypoints"][0], kpts_bg["body"]["keypoints"][0]]
            # 0=上传图符合要求, 1=图1或图2中未检测到上半身关键点, 2=图1和图2未能同时检测到/检测不到下半身关键点
            if upper_mote or upper_bg:
                return [1]
            elif lower_mote != lower_bg:
                return [2]
            elif lower_mote == 0:
                return [0, [mote, bg], "full", kpts]
            return [0, [mote, bg], "upper", kpts]
        else:  # 背景图是内置背景图
            # 0=上传模特图符合要求, 1=模特图中未检测到上半身关键点
            if upper_mote:
                return [1]
            elif lower_mote:
                # bg_lst: 0-full 1-upper
                return [0, [mote, bg_lst[0]], "full", kpts]
            return [0, [mote, bg_lst[1]], "upper", kpts]


style_config = {
    "european_girl":
        {"european_girl_sweet": {"lora_model_path": ["models/mote/sweet0_eu_female0_1024x1024_10epoches.safetensors",
                                                     "models/mote/sweet2_eu_female2_1024x1024_less_makeup_10epoches.safetensors"],
                                 "ratio": [0.2, 0.2]},  # 甜美
         # "european_girl_neutral": {"lora_model_path": ["models/mote/neutral0_eu_female0_1024×1024_less_makeup_10epoches.safetensors", "models/mote/neutral1_eu_female1_1024×1024_less_makeup_10epoches.safetensors"], "ratio": [0.4, 0.4]},     # 中性
         "european_girl_elegance": {
             "lora_model_path": ["models/mote/grace0_eu_female0_1024x1024_less_makeup_10epoches.safetensors",
                                 "models/mote/grace1_eu_female1_1024x1024_less_makeup_10epoches.safetensors"],
             "ratio": [0.2, 0.6]},  # 优雅
         "european_girl_nature": {
             "lora_model_path": ["models/mote/nature0_eu_female0_1024x1024_less_makeup_10epoches.safetensors",
                                 "models/mote/nature1_eu_female1_1024x1024_less_makeup_10epoches.safetensors"],
             "ratio": [0.5, 0.3]},  # 自然
         "european_girl_vanguard": {
             "lora_model_path": ["models/mote/progress0_eu_female0_1024x1024_less_makeup_10epoches.safetensors",
                                 "models/mote/progress1_eu_female1_1024x1024_10epoches.safetensors"],
             "ratio": [0.4, 0.4]},  # 前卫
         "european_girl_romantic": {
             "lora_model_path": ["models/mote/romantic0_eu_female0_1024x1024_less_makeup_10epoches.safetensors",
                                 "models/mote/romantic1_eu_female1_1024x1024_less_makeup_10epoches.safetensors"],
             "ratio": [0.6, 0.2]},  # 浪漫
         "european_girl_noble": {"lora_model_path": ["models/mote/noble0_eu_female0_1024x1024_10epoches.safetensors",
                                                     "models/mote/noble1_eu_female1_1024x1024_less_makeup_10epoches.safetensors"],
                                 "ratio": [0.4, 0.4]}},  # 高贵

    "european_boy":
        {"european_boy_pretty": {"lora_model_path": ["models/mote/handsome2_eu_male2_1024x1024_10epoches.safetensors",
                                                     "models/mote/handsome4_eu_male4_1024x1024_10epoches.safetensors"],
                                 "ratio": [0.2, 0.2]},  # 俊秀
         "european_boy_new": {"lora_model_path": ["models/mote/fashion0_eu_male0_1024x1024_10epoches.safetensors",
                                                  "models/mote/fashion2_eu_male2_1024x1024_10epoches.safetensors"],
                              "ratio": [0.4, 0.4]},  # 新潮
         "european_boy_elegance": {"lora_model_path": ["models/mote/gentle0_eu_male0_1024x1024_10epoches.safetensors",
                                                       "models/mote/gentle1_eu_male1_1024x1024_10epoches.safetensors"],
                                   "ratio": [0.4, 0.4]},  # 儒雅
         "european_boy_nature": {"lora_model_path": ["models/mote/nature2_eu_male2_1024x1024_10epoches.safetensors",
                                                     "models/mote/nature4_eu_male4_1024x1024_10epoches.safetensors"],
                                 "ratio": [0.4, 0.4]},  # 自然
         "european_boy_newandvig": {"lora_model_path": ["models/mote/sharp0_eu_male0_1024x1024_10epoches.safetensors",
                                                        "models/mote/sharp1_eu_male1_1024x1024_10epoches.safetensors"],
                                    "ratio": [0.4, 0.4]},  # 新锐
         "european_boy_romantic": {"lora_model_path": ["models/mote/romantic0_eu_male0_1024x1024_10epoches.safetensors",
                                                       "models/mote/romantic1_eu_male1_1024x1024_10epoches.safetensors"],
                                   "ratio": [0.6, 0.2]},  # 浪漫
         "european_boy_arrogance": {
             "lora_model_path": ["models/mote/domineer2_eu_male2_1024x1024_10epoches.safetensors",
                                 "models/mote/domineer4_eu_male4_1024x1024_10epoches.safetensors"],
             "ratio": [0.4, 0.4]}},  # 霸气

    "asian_girl":
        {"asian_girl_sweet": {
            "lora_model_path": ["models/mote/sweet3_as_female3_1024x1024_less_makeup_10epoches.safetensors",
                                "models/mote/sweet4_as_female4_1024x1024_less_makeup_10epoches.safetensors"],
            "ratio": [0.6, 0.4]},  # 甜美
         "asian_girl_neutral": {
             "lora_model_path": ["models/mote/neutral0_as_female0_1024x1024_less_makeup_10epoches.safetensors",
                                 "models/mote/neutral1_as_female1_1024x1024_less_makeup_10epoches.safetensors"],
             "ratio": [0.4, 0.4]},  # 中性
         "asian_girl_elegance": {
             "lora_model_path": ["models/mote/grace0_as_female0_1024x1024_less_makeup_10epoches.safetensors",
                                 "models/mote/grace1_as_female1_1024x1024_less_makeup_10epoches.safetensors"],
             "ratio": [0.5, 0.3]},  # 优雅
         "asian_girl_nature": {
             "lora_model_path": ["models/mote/nature0_as_female0_1024x1024_less_makeup_10epoches.safetensors",
                                 "models/mote/nature1_as_female1_1024x1024_less_makeup_10epoches.safetensors"],
             "ratio": [0.4, 0.4]},  # 自然
         "asian_girl_vanguard": {
             "lora_model_path": ["models/mote/progress0_as_female0_1024x1024_less_makeup_10epoches.safetensors",
                                 "models/mote/progress1_as_female1_1024x1024_less_makeup_10epoches.safetensors"],
             "ratio": [0.4, 0.4]},  # 前卫
         "asian_girl_romantic": {
             "lora_model_path": ["models/mote/romantic0_as_female0_1024x1024_less_makeup_10epoches.safetensors",
                                 "models/mote/romantic1_as_female1_1024x1024_less_makeup_10epoches.safetensors"],
             "ratio": [0.4, 0.4]},  # 浪漫
         "asian_girl_noble": {"lora_model_path": ["models/mote/noble0_as_female0_1024x1024_10epoches.safetensors",
                                                  "models/mote/noble1_as_female1_1024x1024_10epoches.safetensors"],
                              "ratio": [0.4, 0.4]}},  # 高贵     # 随机

    "asian_boy":
        {"asian_boy_pretty": {"lora_model_path": ["models/mote/handsome2_as_male2_1024x1024_10epoches.safetensors",
                                                  "models/mote/handsome3_as_male3_1024x1024_10epoches.safetensors"],
                              "ratio": [0.4, 0.4]},  # 俊秀
         "asian_boy_new": {"lora_model_path": ["models/mote/fashion0_as_male0_1024x1024_10epoches.safetensors",
                                               "models/mote/fashion1_as_male1_1024x1024_10epoches.safetensors"],
                           "ratio": [0.4, 0.4]},  # 新潮
         "asian_boy_elegance": {"lora_model_path": ["models/mote/gentle0_as_male0_1024x1024_10epoches.safetensors",
                                                    "models/mote/gentle2_as_male2_1024x1024_10epoches.safetensors"],
                                "ratio": [0.4, 0.4]},  # 儒雅
         "asian_boy_nature": {"lora_model_path": ["models/mote/nature0_as_male0_1024x1024_10epoches.safetensors",
                                                  "models/mote/nature1_as_male1_1024x1024_10epoches.safetensors"],
                              "ratio": [0.4, 0.4]},  # 自然
         "asian_boy_newandvig": {"lora_model_path": ["models/mote/sharp0_as_male0_1024x1024_10epoches.safetensors",
                                                     "models/mote/sharp1_as_male1_1024x1024_10epoches.safetensors"],
                                 "ratio": [0.2, 0.6]},  # 新锐
         "asian_boy_romantic": {"lora_model_path": ["models/mote/romantic0_as_male0_1024x1024_10epoches.safetensors",
                                                    "models/mote/romantic1_as_male1_1024x1024_10epoches.safetensors"],
                                "ratio": [0.6, 0.2]},  # 浪漫
         "asian_boy_arrogance": {"lora_model_path": ["models/mote/domineer0_as_male0_1024x1024_10epoches.safetensors",
                                                     "models/mote/domineer1_as_male1_1024x1024_10epoches.safetensors"],
                                 "ratio": [0.4, 0.4]}},  # 霸气
}

mote_config = {
    "asian_boy": {"base_path": "models/mote/braBeautifulRealistic_v3",
                  "prompt": "((RAW)), analog style, A stunning portrait of a man. he should smile and have  a captivating gaze, and an alluring expression. The background should be neutral and simple, allowing the man to be the image's focal point. Please pay close attention to the details, such as the highlights and shadows on his face and hair, to create a lifelike and realistic image. Experiment with different lighting techniques to enhance the mood and atmosphere of the portrait. The final product should be a masterpiece that captures the essence and beauty of the man, ((highly detailed skin, skin details)), sharp focus, volumetric fog, 8k UHD, DSLR, high quality, film grain, Fujifilm XT3 (black hair)",
                  "negative_prompt": "ng_deepnegative_v1_75t, paintings, sketches, (worst quality, low quality, normal quality:2), lowres, ((monochrome, grayscale)), skin spots, acnes, skin blemishes, age spot, backlight, ugly, Overexposure,(((tilted head))),((cut off))",
                  "seed": None},
    "asian_girl": {"base_path": "models/mote/braBeautifulRealistic_v3",
                   "prompt": "((RAW)), analog style, A stunning portrait of a young girl. She should smile and have  a captivating gaze, and an alluring expression. The background should be neutral and simple, allowing the woman to be the image's focal point. Please pay close attention to the details, such as the highlights and shadows on her face and hair, to create a lifelike and realistic image. Experiment with different lighting techniques to enhance the mood and atmosphere of the portrait. The final product should be a masterpiece that captures the essence and beauty of the woman, ((highly detailed skin, skin details)), sharp focus, volumetric fog, 8k UHD, DSLR, high quality, film grain, Fujifilm XT3 (black hair)",
                   "negative_prompt": "ng_deepnegative_v1_75t, (badhandv4), (worst quality:2), (low quality:2), (normal quality:2), lowres, bad anatomy, bad hands, ((monochrome)), ((grayscale)) watermark, (moles:2)",
                   "seed": None},
    "european_boy": {"base_path": "models/mote/CyberRealistic_V3",
                     "prompt": "((photorealism)),(4k textures),natural skin texture, 24mm, 4k textures, soft cinematic light, RAW photo, photorealism, photorealistic, intricate, elegant, highly detailed, sharp focus, A stunning portrait of a young man. he should smile and have  a captivating gaze, and an alluring expression. The background should be neutral and simple, allowing the man to be the image's focal point. Please pay close attention to the details, to create a lifelike and realistic image.  ((masterpiece)), ((highly detailed skin, skin details)), sharp focus,",
                     "negative_prompt": "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), (depth of field:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck",
                     "seed": None},
    "european_girl": {"base_path": "models/mote/CyberRealistic_V3",
                      "prompt": "((RAW)), analog style, A stunning portrait of a young girl. Blue eyes. She should smlie and  have  a captivating gaze, and an alluring expression. The background should be neutral and simple, allowing the woman to be the image's focal point. Please pay close attention to the details, such as the highlights and shadows on her face and hair, to create a lifelike and realistic image. Experiment with different lighting techniques to enhance the mood and atmosphere of the portrait. The final product should be a masterpiece that captures the essence and beauty of the woman, ((highly detailed skin, skin details)), sharp focus",
                      "negative_prompt": "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck",
                      "seed": None},
}


# 换脸
class FaceInpainter():
    def __init__(self, mote_key, style_key):
        self.mote_config = mote_config
        self.style_config = style_config
        self.prompt = self.mote_config[mote_key]["prompt"]
        self.negative_prompt = self.mote_config[mote_key]["negative_prompt"]
        base_path = self.mote_config[mote_key]["base_path"]
        controlnet = MultiControlNetModel([pose_model])
        self.pipe_control = VIPStableDiffusionControlNetInpaintPipeline.from_pretrained(base_path,
                                                                                        controlnet=controlnet,
                                                                                        torch_dtype=dtype).to(device)
        if "random" in style_key:
            style_key = random.choice(list(self.style_config[mote_key]))
        for i in range(len(self.style_config[mote_key][style_key]["lora_model_path"])):
            lora_model_path = self.style_config[mote_key][style_key]["lora_model_path"][i]
            lora_ratio = self.style_config[mote_key][style_key]["ratio"][i]
            self.pipe_control = load_lora_weights(self.pipe_control, lora_model_path, lora_ratio, device="cuda",
                                                  dtype=dtype)
        self.pipe_control.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe_control.scheduler.config)
        load_webui_textual_inversion("embedding", self.pipe_control)

    def __call__(self, mote, seed=-1, **kwargs):
        pose_image = pose_inferencer(np.array(mote), hand=False)
        pose_image = Image.fromarray(pose_image)

        facedet_res = mediapipe_face_detection(mote)
        bboxes, masks, preview = facedet_res
        if masks is None:
            return mote
        mask = mask_process(masks[0], invert_mask=False)
        crop_region = get_crop_region(np.array(mask), pad=256)
        crop_region = expand_crop_region(crop_region, input_size[1], input_size[0], mask.width, mask.height)
        mask = mask.crop(crop_region)
        img = mote.crop(crop_region)

        print(f'processing seed = {seed}')
        generator = get_torch_generator(seed, device=device)
        pipe_list, image_overlay = self.pipe_control(
            prompt=self.prompt,
            negative_prompt=self.negative_prompt,
            image=img,
            mask_image=mask,
            control_image=[pose_image],
            height=input_size[0],
            width=input_size[1],
            strength=0.4,
            num_inference_steps=28,
            guidance_scale=6,
            num_images_per_prompt=1,
            generator=generator,
            controlnet_conditioning_scale=0.)
        pipe_list = apply_codeformer(face_restored_path, pipe_list)
        pipe_list = alpha_composite(pipe_list, image_overlay)
        x1, y1, x2, y2 = crop_region
        res = list()
        for i, out_img in enumerate(pipe_list):
            res_image = mote.copy()
            res_image.paste(out_img.resize((int(x2 - x1), int(y2 - y1)), resample=Image.LANCZOS), (x1, y1))
            res.append(res_image)
        return res[0]


scene_config = {
    "art_gallery": {
        "prompt": "(bokeh:0.8),liminalspaces, An empty art gallery, a silent tribute to artistic expression, beautiful light, best quality, masterpiece, ultra highres,  very detailed skin, 4k, photorealistic, masterpiece, very high detailed",
        "negative_prompt": "ng_deepnegative_v1_75t , (badhandv4), verybadimagenegative_v1.3, easynegative, EasyNegative, (worst quality:2), (low quality:2), (normal quality:2), lowres, ((monochrome)), ((grayscale)), 2 body, 2 pussy, 2 upper, 2 lower, ((2 head)), ((3 hands)), 3 feet, bad anatomy,((cross-eyed)), bad hands, bad feet, watermark, (moles:2), illustration ,3d, sepia ,painting ,cartoon, sketch, light above the head, stand on the table, high light on body,(people),beam,((tilt)),(Complex Background),(Chaotic background),(Sloping background), ((light at background)),((cloak)),((no peoples background)),((backlight)), ((light above the head)), high light on body, highly saturated colors, (((sunset:1.1))), (((sun))),ng_deepnegative_v1_75t , (badhandv4), verybadimagenegative_v1.3, easynegative, EasyNegative, (worst quality:2), (low quality:2), (normal quality:2), lowres, ((monochrome)), ((grayscale)), 2 body, 2 pussy, 2 upper, 2 lower, ((2 head)), ((3 hands)), 3 feet, bad anatomy,((cross-eyed)), bad hands, bad feet, watermark, (moles:2), illustration ,3d, sepia ,painting ,cartoon, sketch, light above the head, stand on the table, high light on body",
        "steps": 30, "cfgscale": 7, "denoising": 0.5, "mask_blur": 4,
        "enhance": ['color', 'contrast'],
        "enhance_guidance_start": 0.1,
        "enhance_guidance_end": 0.5,
        "enhance_scale": [0.025, 0.005]},
    "cafe": {
        "prompt": "((standing on the clear floor)), coffee cup on table, dining table, beautiful light, best quality, masterpiece, ultra highres,  very detailed skin, 4k, photorealistic, masterpiece, very high detailed",
        "negative_prompt": "ng_deepnegative_v1_75t , (badhandv4), verybadimagenegative_v1.3, easynegative, EasyNegative, (worst quality:2), (low quality:2), (normal quality:2), lowres, ((monochrome)), ((grayscale)), 2 body, 2 pussy, 2 upper, 2 lower, ((2 head)), ((3 hands)), 3 feet, bad anatomy,((cross-eyed)), bad hands, bad feet, watermark, (moles:2), illustration ,3d, sepia ,painting ,cartoon, sketch, light above the head, stand on the table, high light on body,(people),beam,((tilt)),(Complex Background),(Chaotic background),(Sloping background), ((light at background)),((cloak)),((no peoples background)),((backlight)), ((light above the head)), high light on body, highly saturated colors, (((sunset:1.1))), (((sun))),ng_deepnegative_v1_75t , (badhandv4), verybadimagenegative_v1.3, easynegative, EasyNegative, (worst quality:2), (low quality:2), (normal quality:2), lowres, ((monochrome)), ((grayscale)), 2 body, 2 pussy, 2 upper, 2 lower, ((2 head)), ((3 hands)), 3 feet, bad anatomy,((cross-eyed)), bad hands, bad feet, watermark, (moles:2), illustration ,3d, sepia ,painting ,cartoon, sketch, light above the head, stand on the table, high light on body",
        "steps": 30, "cfgscale": 7, "denoising": 0.5, "mask_blur": 4,
        "enhance": ['color', 'contrast'],
        "enhance_guidance_start": 0.1,
        "enhance_guidance_end": 0.5,
        "enhance_scale": [0.01, 0.005]},
    "home": {
        "prompt": "((standing on the clear floor)), window, photorealistic((living room:1.4)), (exquisite decoration),beautiful light, best quality, masterpiece, ultra highres,  very detailed skin, 4k, photorealistic, masterpiece, very high detailed",
        "negative_prompt": "stand on the sofa, 2 sofa, ng_deepnegative_v1_75t , (badhandv4), verybadimagenegative_v1.3, easynegative, EasyNegative, (worst quality:2), (low quality:2), (normal quality:2), lowres, ((monochrome)), ((grayscale)), 2 body, 2 pussy, 2 upper, 2 lower, ((2 head)), ((3 hands)), 3 feet, bad anatomy,((cross-eyed)), bad hands, bad feet, watermark, (moles:2), illustration ,3d, sepia ,painting ,cartoon, sketch, light above the head, stand on the table, high light on body,(people),beam,((tilt)),(Complex Background),(Chaotic background),(Sloping background), ((light at background)),((cloak)),((no peoples background)),((backlight)), ((light above the head)), high light on body, highly saturated colors, (((sunset:1.1))), (((sun))),ng_deepnegative_v1_75t , (badhandv4), verybadimagenegative_v1.3, easynegative, EasyNegative, (worst quality:2), (low quality:2), (normal quality:2), lowres, ((monochrome)), ((grayscale)), 2 body, 2 pussy, 2 upper, 2 lower, ((2 head)), ((3 hands)), 3 feet, bad anatomy,((cross-eyed)), bad hands, bad feet, watermark, (moles:2), illustration ,3d, sepia ,painting ,cartoon, sketch, light above the head, stand on the table, high light on body",
        "steps": 30, "cfgscale": 7, "denoising": 0.5, "mask_blur": 4,
        "enhance": ['color', 'contrast'],
        "enhance_guidance_start": 0.1,
        "enhance_guidance_end": 0.5,
        "enhance_scale": [0.01, 0.005]},
    "office": {
        "prompt": "(simple background), (stand on the floor), (clear white Minimalism office:1.4),clear window background, indoors, city landscape, computer,beautiful light, best quality, masterpiece, ultra highres,  very detailed skin, 4k, photorealistic, masterpiece, very high detailed",
        "negative_prompt": "((stand on the sofa)),((stand on the table)), ng_deepnegative_v1_75t , (badhandv4), verybadimagenegative_v1.3, easynegative, EasyNegative, (worst quality:2), (low quality:2), (normal quality:2), lowres, ((monochrome)), ((grayscale)), 2 body, 2 pussy, 2 upper, 2 lower, ((2 head)), ((3 hands)), 3 feet, bad anatomy,((cross-eyed)), bad hands, bad feet, watermark, (moles:2), illustration ,3d, sepia ,painting ,cartoon, sketch, light above the head, stand on the table, high light on body,(people),beam,((tilt)),(Complex Background),(Chaotic background),(Sloping background), ((light at background)),((cloak)),((no peoples background)),((backlight)), ((light above the head)), high light on body, highly saturated colors, (((sunset:1.1))), (((sun))),ng_deepnegative_v1_75t , (badhandv4), verybadimagenegative_v1.3, easynegative, EasyNegative, (worst quality:2), (low quality:2), (normal quality:2), lowres, ((monochrome)), ((grayscale)), 2 body, 2 pussy, 2 upper, 2 lower, ((2 head)), ((3 hands)), 3 feet, bad anatomy,((cross-eyed)), bad hands, bad feet, watermark, (moles:2), illustration ,3d, sepia ,painting ,cartoon, sketch, light above the head, stand on the table, high light on body",
        "steps": 30, "cfgscale": 7, "denoising": 0.5, "mask_blur": 4,
        "enhance": ['color', 'contrast'],
        "enhance_guidance_start": 0.1,
        "enhance_guidance_end": 0.5,
        "enhance_scale": [0.01, 0.005]},
    "office_building": {
        "prompt": "bokeh,outdoor, Simple architecture,Minimalist architecture, [glass exterior wall], clear floor, beautiful light, best quality, masterpiece, ultra highres,  very detailed skin, 4k, photorealistic, masterpiece, very high detailed",
        "negative_prompt": "ng_deepnegative_v1_75t , (badhandv4), verybadimagenegative_v1.3, easynegative, EasyNegative, (worst quality:2), (low quality:2), (normal quality:2), lowres, ((monochrome)), ((grayscale)), 2 body, 2 pussy, 2 upper, 2 lower, ((2 head)), ((3 hands)), 3 feet, bad anatomy,((cross-eyed)), bad hands, bad feet, watermark, (moles:2), illustration ,3d, sepia ,painting ,cartoon, sketch, light above the head, stand on the table, high light on body,(people),beam,((tilt)),(Complex Background),(Chaotic background),(Sloping background), ((light at background)),((cloak)),((no peoples background)),((backlight)), ((light above the head)), high light on body, highly saturated colors, (((sunset:1.1))), (((sun))),ng_deepnegative_v1_75t , (badhandv4), verybadimagenegative_v1.3, easynegative, EasyNegative, (worst quality:2), (low quality:2), (normal quality:2), lowres, ((monochrome)), ((grayscale)), 2 body, 2 pussy, 2 upper, 2 lower, ((2 head)), ((3 hands)), 3 feet, bad anatomy,((cross-eyed)), bad hands, bad feet, watermark, (moles:2), illustration ,3d, sepia ,painting ,cartoon, sketch, light above the head, stand on the table, high light on body",
        "steps": 30, "cfgscale": 7, "denoising": 0.5, "mask_blur": 4,
        "enhance": ['color', 'contrast'],
        "enhance_guidance_start": 0.1,
        "enhance_guidance_end": 0.5,
        "enhance_scale": [0.01, 0.005]},
    "park": {
        "prompt": "photorealistic (garden background:1.4), Standing on the ground,scenery, tree, outdoors, grass, nature, road, forest, path,beautiful light, best quality, masterpiece, ultra highres,  very detailed skin, 4k, photorealistic, masterpiece, very high detailed",
        "negative_prompt": "((stand on water)), ((the waves)), ((cloak)), mountains, ((reefs)), ((backlight)), ((light above the head)), stand on the table, high light on body, highly saturated colors, (((sunset:1.1))), (((sun))),ng_deepnegative_v1_75t , (badhandv4), verybadimagenegative_v1.3, easynegative, EasyNegative, (worst quality:2), (low quality:2), (normal quality:2), lowres, ((monochrome)), ((grayscale)), 2 body, 2 pussy, 2 upper, 2 lower, ((2 head)), ((3 hands)), 3 feet, bad anatomy,((cross-eyed)), bad hands, bad feet, watermark, (moles:2), illustration ,3d, sepia ,painting ,cartoon, sketch, light above the head, stand on the table, high light on body,(people),beam,((tilt)),(Complex Background),(Chaotic background),(Sloping background), ((light at background)),((cloak)),((no peoples background)),((backlight)), ((light above the head)), high light on body, highly saturated colors, (((sunset:1.1))), (((sun))),ng_deepnegative_v1_75t , (badhandv4), verybadimagenegative_v1.3, easynegative, EasyNegative, (worst quality:2), (low quality:2), (normal quality:2), lowres, ((monochrome)), ((grayscale)), 2 body, 2 pussy, 2 upper, 2 lower, ((2 head)), ((3 hands)), 3 feet, bad anatomy,((cross-eyed)), bad hands, bad feet, watermark, (moles:2), illustration ,3d, sepia ,painting ,cartoon, sketch, light above the head, stand on the table, high light on body",
        "steps": 30, "cfgscale": 7, "denoising": 0.5, "mask_blur": 4,
        "enhance": ['sharpness', 'contrast'],
        "enhance_guidance_start": 0.1,
        "enhance_guidance_end": 0.5,
        "enhance_scale": [0.025, 0.005]},
    "seaside": {
        "prompt": "(beautiful beach background:1.4), blue sky, ocean and beach, ((standing on beach)),beautiful light, best quality, masterpiece, ultra highres,  very detailed skin, 4k, photorealistic, masterpiece, very high detailed",
        "negative_prompt": "ng_deepnegative_v1_75t , (badhandv4), verybadimagenegative_v1.3, easynegative, EasyNegative, (worst quality:2), (low quality:2), (normal quality:2), lowres, ((monochrome)), ((grayscale)), 2 body, 2 pussy, 2 upper, 2 lower, ((2 head)), ((3 hands)), 3 feet, bad anatomy,((cross-eyed)), bad hands, bad feet, watermark, (moles:2), illustration ,3d, sepia ,painting ,cartoon, sketch, light above the head, stand on the table, high light on body,(people),beam,((tilt)),(Complex Background),(Chaotic background),(Sloping background), ((light at background)),((cloak)),((no peoples background)),((backlight)), ((light above the head)), high light on body, highly saturated colors, (((sunset:1.1))), (((sun))),ng_deepnegative_v1_75t , (badhandv4), verybadimagenegative_v1.3, easynegative, EasyNegative, (worst quality:2), (low quality:2), (normal quality:2), lowres, ((monochrome)), ((grayscale)), 2 body, 2 pussy, 2 upper, 2 lower, ((2 head)), ((3 hands)), 3 feet, bad anatomy,((cross-eyed)), bad hands, bad feet, watermark, (moles:2), illustration ,3d, sepia ,painting ,cartoon, sketch, light above the head, stand on the table, high light on body",
        "steps": 30, "cfgscale": 7, "denoising": 0.5, "mask_blur": 4,
        "enhance": ['sharpness', 'contrast'],
        "enhance_guidance_start": 0.1,
        "enhance_guidance_end": 0.5,
        "enhance_scale": [0.015, 0.003]},
    "street_view": {
        "prompt": "photorealistic (street:1.5),(streetscenervy), road, path, cleaning, white, (Perspective), telephotolens,beautiful light, best quality, masterpiece, ultra highres,  very detailed skin, 4k, photorealistic, masterpiece, very high detailed",
        "negative_prompt": "ng_deepnegative_v1_75t , (badhandv4), verybadimagenegative_v1.3, easynegative, EasyNegative, (worst quality:2), (low quality:2), (normal quality:2), lowres, ((monochrome)), ((grayscale)), 2 body, 2 pussy, 2 upper, 2 lower, ((2 head)), ((3 hands)), 3 feet, bad anatomy,((cross-eyed)), bad hands, bad feet, watermark, (moles:2), illustration ,3d, sepia ,painting ,cartoon, sketch, light above the head, stand on the table, high light on body,(people),beam,((tilt)),(Complex Background),(Chaotic background),(Sloping background), ((light at background)),((cloak)),((no peoples background)),((backlight)), ((light above the head)), high light on body, highly saturated colors, (((sunset:1.1))), (((sun))),ng_deepnegative_v1_75t , (badhandv4), verybadimagenegative_v1.3, easynegative, EasyNegative, (worst quality:2), (low quality:2), (normal quality:2), lowres, ((monochrome)), ((grayscale)), 2 body, 2 pussy, 2 upper, 2 lower, ((2 head)), ((3 hands)), 3 feet, bad anatomy,((cross-eyed)), bad hands, bad feet, watermark, (moles:2), illustration ,3d, sepia ,painting ,cartoon, sketch, light above the head, stand on the table, high light on body",
        "steps": 30, "cfgscale": 7, "denoising": 0.5, "mask_blur": 4,
        "enhance": ['sharpness', 'contrast'],
        "enhance_guidance_start": 0.1,
        "enhance_guidance_end": 0.5,
        "enhance_scale": [0.025, 0.005]},
    "studio": {
        "prompt": "photorealistic (clear grey background:1.4),beautiful light, best quality, masterpiece, ultra highres,  very detailed skin, 4k, photorealistic, masterpiece, very high detailed",
        "negative_prompt": "ng_deepnegative_v1_75t , (badhandv4), verybadimagenegative_v1.3, easynegative, EasyNegative, (worst quality:2), (low quality:2), (normal quality:2), lowres, ((monochrome)), ((grayscale)), 2 body, 2 pussy, 2 upper, 2 lower, ((2 head)), ((3 hands)), 3 feet, bad anatomy,((cross-eyed)), bad hands, bad feet, watermark, (moles:2), illustration ,3d, sepia ,painting ,cartoon, sketch, light above the head, stand on the table, high light on body,(people),beam,((tilt)),(Complex Background),(Chaotic background),(Sloping background), ((light at background)),((cloak)),((no peoples background)),((backlight)), ((light above the head)), high light on body, highly saturated colors, (((sunset:1.1))), (((sun))),ng_deepnegative_v1_75t , (badhandv4), verybadimagenegative_v1.3, easynegative, EasyNegative, (worst quality:2), (low quality:2), (normal quality:2), lowres, ((monochrome)), ((grayscale)), 2 body, 2 pussy, 2 upper, 2 lower, ((2 head)), ((3 hands)), 3 feet, bad anatomy,((cross-eyed)), bad hands, bad feet, watermark, (moles:2), illustration ,3d, sepia ,painting ,cartoon, sketch, light above the head, stand on the table, high light on body",
        "steps": 30, "cfgscale": 7, "denoising": 0.5, "mask_blur": 4,
        "enhance": None,
        "enhance_guidance_start": 0.1,
        "enhance_guidance_end": 0.5,
        "enhance_scale": [0.003]},
    "wall": {
        "prompt": "bokeh,clear background,next to the  wall,physically-based rendering, beautiful light, best quality, masterpiece, ultra highres,  very detailed skin, 4k, photorealistic, masterpiece, very high detailed",
        "negative_prompt": "ng_deepnegative_v1_75t , (badhandv4), verybadimagenegative_v1.3, easynegative, EasyNegative, (worst quality:2), (low quality:2), (normal quality:2), lowres, ((monochrome)), ((grayscale)), 2 body, 2 pussy, 2 upper, 2 lower, ((2 head)), ((3 hands)), 3 feet, bad anatomy,((cross-eyed)), bad hands, bad feet, watermark, (moles:2), illustration ,3d, sepia ,painting ,cartoon, sketch, light above the head, stand on the table, high light on body,(people),beam,((tilt)),(Complex Background),(Chaotic background),(Sloping background), ((light at background)),((cloak)),((no peoples background)),((backlight)), ((light above the head)), high light on body, highly saturated colors, (((sunset:1.1))), (((sun))),ng_deepnegative_v1_75t , (badhandv4), verybadimagenegative_v1.3, easynegative, EasyNegative, (worst quality:2), (low quality:2), (normal quality:2), lowres, ((monochrome)), ((grayscale)), 2 body, 2 pussy, 2 upper, 2 lower, ((2 head)), ((3 hands)), 3 feet, bad anatomy,((cross-eyed)), bad hands, bad feet, watermark, (moles:2), illustration ,3d, sepia ,painting ,cartoon, sketch, light above the head, stand on the table, high light on body",
        "steps": 30, "cfgscale": 7, "denoising": 0.5, "mask_blur": 4,
        "enhance": None,
        "enhance_guidance_start": 0.1,
        "enhance_guidance_end": 0.5,
        "enhance_scale": [0.003]},
}


# 换背景
class BgInpainter():
    def __init__(self, load_lora=False):
        self.scene_config = scene_config
        base_path = "models/mote/CyberRealistic_V3"
        controlnet = [canny_model, depth_model, pose_model] #, pose_model, light_model
        self.pipe_control = VIPStableDiffusionControlNetInpaintPipeline.from_pretrained(
            base_path, controlnet=controlnet, torch_dtype=torch.float16).to('cuda')
        self.pipe_control.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe_control.scheduler.config)
        load_webui_textual_inversion("embedding", self.pipe_control)
        # self.pipe_control.load_lora_weights("sd-model-finetuned-lora", weight_name="pytorch_lora_weights.safetensors")
        if load_lora:
            for i in range(len(style_config["european_girl"]["european_girl_sweet"]["lora_model_path"])):
                lora_model_path = style_config["european_girl"]["european_girl_sweet"]["lora_model_path"][i]
                lora_ratio = style_config["european_girl"]["european_girl_sweet"]["ratio"][i]
                self.pipe_control = load_lora_weights(self.pipe_control, lora_model_path, lora_ratio, device="cuda",
                                                    dtype=dtype)


        self.get_depth = MidasDetector()

    def __call__(self, scene_key, mote, ori_mask, seed, hair=True, **kwargs):
        # mote_config['european_girl']['prompt'] +
        prompt = self.scene_config[scene_key]["prompt"]
        negative_prompt = self.scene_config[scene_key]["negative_prompt"]

        # pose_image = pose_inferencer(np.array(mote), hand=True)
        # pose_image = Image.fromarray(pose_image)

        # label_parsing = human_parsing.inference(mote.copy()).astype("uint8")
        # face_mask = np.zeros_like(label_parsing)
        # face_mask[label_parsing == 10] = 255
        # face_mask[label_parsing == 13] = 255
        # face_mask = cv2.dilate(face_mask, np.ones((5, 5), np.uint8), iterations=1)

        ori_mask = np.array(ori_mask.convert("L"))
        fg_mask = cv2.threshold(ori_mask, 127, 1, cv2.THRESH_BINARY)[1]
        canny_image = cv2.Canny(np.array(mote) * fg_mask[..., None], 100, 200)
        canny_image = np.tile(canny_image[..., None], [1, 1, 3])
        canny_image = Image.fromarray(canny_image)

        # canny_image = cv2.Canny(ori_mask, 100, 200)
        # canny_image = np.tile(canny_image[..., None], [1, 1, 3])
        # canny_image = Image.fromarray(canny_image)

        if hair:
            label_parsing = human_parsing.inference(mote.copy()).astype('uint8')
            hair_mask = np.zeros_like(label_parsing)
            hair_mask[label_parsing == 2] = 255
            ori_mask[ori_mask < 128] = 0
            ori_mask[hair_mask == 255] = 0
        # clo_mask = clotheseg_inferencer(mote)
        # mask_image = mask_process(clo_mask, invert_mask=True)
        # tmp = np.array(mask_image.convert("RGB"))

        mask_image = mask_process(Image.fromarray(ori_mask), invert_mask=True)
        tmp = np.asarray(mask_image.convert("RGB"))

        depth = self.get_depth(mote)
        depth[tmp > 127.5] = 0
        depth_image = Image.fromarray(depth)

        enhance = kwargs.get("enhance", None)
        enhance_guidance_start = kwargs.get("enhance_guidance_start", None)
        enhance_guidance_end = kwargs.get("enhance_guidance_end", None)
        enhance_scale = kwargs.get("enhance_scale", None)
        if enhance is not None and isinstance(enhance, list):
            enhance = kwargs["enhance"]
            enhance_guidance_start = kwargs["enhance_guidance_start"]
            enhance_guidance_end = kwargs["enhance_guidance_end"]
            enhance_scale = kwargs["enhance_scale"]
        elif enhance is not None:
            enhance = self.scene_config[scene_key]["enhance"]
            enhance_guidance_start = self.scene_config[scene_key]["enhance_guidance_start"]
            enhance_guidance_end = self.scene_config[scene_key]["enhance_guidance_end"]
            enhance_scale = self.scene_config[scene_key]["enhance_scale"]

        print(f'processing seed = {seed}')
        generator = get_torch_generator(seed, device=device)
        pipe_list, image_overlay = self.pipe_control(
            prompt="(black or brown hair: 1.2)" + prompt,
            negative_prompt=negative_prompt, # "(white hair, green hair, grey hair)" +
            image=mote,
            mask_image=mask_image,
            control_image=[canny_image, depth_image, depth_image],
            height=input_size[1],
            width=input_size[0],
            strength=self.scene_config[scene_key]['denoising'],
            num_inference_steps=self.scene_config[scene_key]['steps'],
            guidance_scale=self.scene_config[scene_key]['cfgscale'],
            num_images_per_prompt=1,
            generator=generator,
            controlnet_conditioning_scale=[1.0, 0.2, 0.0],
            control_guidance_end=1.0,

            # cross_attention_kwargs={'scale': 0.2},

            enhance=enhance,
            enhance_guidance_start=enhance_guidance_start,
            enhance_guidance_end=enhance_guidance_end,
            enhance_scale=enhance_scale)
        pipe_list = apply_codeformer(face_restored_path, pipe_list)
        pipe_list = alpha_composite(pipe_list, image_overlay)
        return pipe_list[0]


class Main():
    def __init__(self, load_lora=False):
        self.data_preprocess = Preprocesser()
        self.bg_inpainter = BgInpainter(load_lora=load_lora)

    def change_mote(self, change_key, input_image, seed=-1, **kwargs):
        if change_key == "change_head":  # 换头
            # head_inpainter = HeadInpainter(mote_key="asian_girl", style_key="asian_girl_sweet")
            # face_inpainter = FaceInpainter(mote_key="asian_girl", style_key="asian_girl_sweet")
            # head_res = head_inpainter(mote=input_image, batch_size=batch_size)
            # face_res = list()
            # for res in head_res:
            #     face_res.append(face_inpainter(mote=res, batch_size=1)[0])
            pass
        else:  # 换脸
            face_res = self.face_inpainter(mote=input_image, seed=seed)
        return face_res

    def change_bg(self, scene_key, input_image, clo_mask, seed, **kwargs):
        return self.bg_inpainter(scene_key=scene_key,
                                mote=input_image,
                                ori_mask=clo_mask,
                                seed=seed, **kwargs)

    def change_all(self, scene_key, input_image, input_bg,
                   kpts, batch_size, use_pih=False, **kwargs):
        out = []
        # clo_mask = Image.open("/xsl/wilson.xu/xsl/mask.png").convert("RGB").resize(input_size)
        # clo_mask = clotheseg_inferencer(input_image)
        # image_overlay = Image.new('RGBa', input_size)
        # image_overlay.paste(input_image.convert("RGBA").convert("RGBa"), mask=clo_mask.convert("L"))
        # image_overlay = image_overlay.convert("RGBA")
        # fixed scene background images
        motemask = postApi_zuotang(input_image).convert("L")
        # motemask = Image.open("mask_bg.png")
        fg_mask = np.asarray(motemask)[..., None].copy()
        fg_mask[fg_mask > 127.5] = 255
        fg_mask[fg_mask <= 127.5] = 0
        fg_mask = fg_mask.astype('bool').astype('uint8')
        # fg_mask = np.array(motemask)[..., None] / 255.0
        img_bg = input_image * fg_mask + input_bg * (1 - fg_mask)
        img_bg = Image.fromarray(np.uint8(img_bg))

        # light_img = img_light_process(input_image)
        # light_img = Image.fromarray(mask * np.array(light_img))
        fg_mask = Image.fromarray(np.uint8(fg_mask[...,0] * 255))
        if use_pih:
            # bg_mask = Image.fromarray((1 - fg_mask)[..., 0] * 255)
            img_pih = pih_inferencer(input_bg, img_bg, fg_mask)
        else:
            img_pih = img_bg

        seed = get_fixed_seed(None)
        for i in range(batch_size):
            res = self.change_bg(scene_key, img_pih, motemask, seed=seed + i, **kwargs)
            # res = self.change_mote(change_key, res, seed=seed + i)
            out.append(res)
        return img_bg, img_pih, out


def hist_match(fg_img, bg_name, bg_val, bins=list(range(257))):
    img = np.array(fg_img)
    L_img = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)[..., 0]
    hist, bin_edges = np.histogram(L_img, bins=bins, density=True)
    func = interp1d(bins[:-1], hist, kind='cubic')
    x = np.linspace(5, 250, num=100)
    y = func(x)
    dis = np.linalg.norm(bg_val - y, axis=1)
    min_name = bg_name[dis.argmin()]
    return min_name


if __name__ == "__main__":
    pipe = Main(load_lora=False)

    save_dir = f'./output_hair'
    os.makedirs(save_dir, exist_ok=True)


    fg_dir = "/xsl/wilson.xu/canny_hair_problem"
    fg_list = os.listdir(fg_dir)

    # bg_dir = "/xsl/wilson.xu/bg"

    bg_scene = pkl_load("/xsl/wilson.xu/bg_scene.pkl")
    bg_hist = pkl_load("/xsl/wilson.xu/bg_hist.pkl")
    bg_name = list(bg_hist['hist_fit'].keys())
    bg_val = np.stack(list(bg_hist['hist_fit'].values()))
    bg_dir = "/xsl/wilson.xu/dataset/tmp_imgs/bg_img"

    for i, name in enumerate(fg_list):
        # if "tryon" not in name:
        #     continue
        # name = "tryon_single_art_gallery_part_ (40).jpg_1_23000505501_2_head.png"
        print(f"{i + 1}/{len(fg_list)}: {name}")
        fg_img = Image.open(join(fg_dir, name)).convert("RGB")
        min_name = hist_match(fg_img, bg_name, bg_val)
        bg_img = Image.open(join(bg_dir, min_name)).convert("RGB").resize(input_size)
        scence = bg_scene[min_name]

        # bg_name = name.split(".")[0]
        # bg_name = "_".join(bg_name.split("_")[2:])
        # scence = "_".join(bg_name.split("_")[:-2])
        # bg_name = join(bg_dir, scence, bg_name.split("_")[-2], bg_name + ".jpg")
        # bg_img = Image.open(bg_name).convert("RGB").resize(input_size)
        bg_lst = [bg_img, bg_img]

        pre_det_res = pipe.data_preprocess(mote=fg_img, bg_lst=bg_lst, is_seedfolder=True)
        if len(pre_det_res) > 1:  # 用户上传图符合要求
            if len(bg_lst) > 0:
                flag, input_images, body_cls, kpts = pre_det_res
                input_mote, input_bg = input_images
            else:
                flag, input_mote = pre_det_res

            img_bg, img_pih, res = pipe.change_all(scene_key=scence,
                                                input_image=input_mote,
                                                input_bg=input_bg,
                                                kpts=kpts, batch_size=1,
                                                enhance=['sharpness', 'contrast'],
                                                enhance_guidance_start=0.1,
                                                enhance_guidance_end=0.5,
                                                enhance_scale=[0.025, 0.005],
                                                )

            out = np.concatenate((input_mote, res[0]), axis=1)
            Image.fromarray(np.uint8(out)).save(f"{save_dir}/{name}.jpg")
            print(f"save image: {save_dir}/{name}.jpg")
        else:  # 用户上传图不符合要求
            flag = pre_det_res[0]






    # fg_img = Image.open("/xsl/wilson.xu/dataset/tmp_imgs/fg_img/45505b41ad30108d38e753e40df555bf.jpg").convert("RGB")
    # bg_dir = "/xsl/wilson.xu/bg"
    # img_dir = "/xsl/wilson.xu/test_0828"
    # bg_list = os.listdir(img_dir)

    # # with open("/xsl/wilson.xu/bg_list.txt", "r") as f:
    # #     bg_list = f.readlines()

    # # for root, dirs, files in os.walk(bg_dir):
    #     # for name in files:
    # for i, name in enumerate(bg_list):
    #     name = name.strip()
    #     # if "studio" not in root:
    #     #     continue
    #     # if name != "seaside_full_ (22).jpg":
    #     #     continue
    #     temp = name.split("_")
    #     scence = "_".join(temp[:-2])
    #     print(f"{scence}: {name}")
    #     bg_img = Image.open(join(bg_dir, scence, temp[-2], name)).convert("RGB").resize(input_size)
    #     bg_lst = [bg_img, bg_img]  # 除了换模特任务，这里必须有输入

    #     pre_det_res = pipe.data_preprocess(mote=fg_img, bg_lst=bg_lst, is_seedfolder=True)
    #     if len(pre_det_res) > 1:  # 用户上传图符合要求
    #         if len(bg_lst) > 0:
    #             flag, input_images, body_cls, kpts = pre_det_res
    #             input_mote, input_bg = input_images
    #         else:
    #             flag, input_mote = pre_det_res

    #         # Step2. 选择各种功能执行
    #         # img_bg, _, res0 = pipe.change_all(scene_key=scence,
    #         #                                     input_image=input_mote,
    #         #                                     input_bg=input_bg,
    #         #                                     kpts=kpts, batch_size=1,
    #         #                                     enhance=None)

    #         img_bg, _, res1 = pipe.change_all(scene_key=scence,
    #                                             input_image=input_mote,
    #                                             input_bg=input_bg,
    #                                             kpts=kpts, batch_size=1,
    #                                             enhance=True)

    #         src_img = Image.open(join(img_dir, name))
    #         out = np.concatenate((src_img, res1[0]), axis=1)
    #         # os.makedirs(join(save_dir, scence), exist_ok=True)
    #         # res1[0].save(f"{save_dir}/{name}")
    #         Image.fromarray(np.uint8(out)).save(f"{save_dir}/{name}")
    #         print(f"save image: {save_dir}/{name}")
    #     else:  # 用户上传图不符合要求
    #         flag = pre_det_res[0]





    # bg_scene = pkl_load("/xsl/wilson.xu/bg_scene.pkl")
    # bg_hist = pkl_load("/xsl/wilson.xu/bg_hist.pkl")
    # bg_name = list(bg_hist['hist_fit'].keys())
    # bg_val = np.stack(list(bg_hist['hist_fit'].values()))
    # fg_dir = "/xsl/wilson.xu/bernshaw_test"
    # bg_dir = "/xsl/wilson.xu/dataset/tmp_imgs/bg_img"

    # fg_list = os.listdir(fg_dir)

    # for i, name in enumerate(fg_list):
    #     print(f"{i + 1}/{len(fg_list)}: {name}")
    #     fg_img = Image.open(join(fg_dir, name)).convert("RGB")
    #     min_name = hist_match(fg_img, bg_name, bg_val)
    #     bg_img = Image.open(join(bg_dir, min_name)).convert("RGB").resize(input_size)
    #     # bg_img = Image.open("/xsl/wilson.xu/bg/seaside/full/seaside_full_ (4).jpg").convert("RGB")
    #     # bg_img = Image.open("/xsl/wilson.xu/bg/street_view/full/street_view_full_ (3).jpg").convert("RGB")
    #     # bg_img = Image.open("/xsl/wilson.xu/dataset/tmp_imgs/bg_img/911e1f4e7d424303e00131ebc3ee8554.jpg").convert("RGB")
    #     bg_lst = [bg_img, bg_img]  # 除了换模特任务，这里必须有输入

    #     pre_det_res = pipe.data_preprocess(mote=fg_img, bg_lst=bg_lst, is_seedfolder=True)
    #     if len(pre_det_res) > 1:  # 用户上传图符合要求
    #         if len(bg_lst) > 0:
    #             flag, input_images, body_cls, kpts = pre_det_res
    #             input_mote, input_bg = input_images
    #         else:
    #             flag, input_mote = pre_det_res

    #         img_bg, img_pih, res = pipe.change_all(scene_key=bg_scene[min_name],
    #                                                 input_image=input_mote,
    #                                                 input_bg=input_bg,
    #                                                 kpts=kpts,
    #                                                 batch_size=2,
    #                                                 enhance=True)

    #         out = np.concatenate((input_mote, img_bg), axis=1)
    #         for j, final_res in enumerate(res):
    #             out = np.concatenate((out, final_res), axis=1)
    #         Image.fromarray(np.uint8(out)).save(f"{save_dir}/{name}")
    #     else:  # 用户上传图不符合要求
    #         flag = pre_det_res[0]
