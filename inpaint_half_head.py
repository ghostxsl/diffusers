import os, cv2
import requests, base64
import numpy as np
from io import BytesIO
from PIL import Image
from tqdm import tqdm
import torch
import random

from diffusers.models import ControlNetModel
from diffusers.pipelines.controlnet.vip_pipeline_controlnet_inpaint import VIPStableDiffusionControlNetInpaintPipeline
from diffusers.schedulers import EulerAncestralDiscreteScheduler, KDPMPP2MDiscreteScheduler
from diffusers.utils.prompt_parser import load_webui_textual_inversion
from diffusers.utils.vip_utils import *
from mmpose.blendpose.inferencer import VIPPoseInferencer
from mmpose.blendpose.utils import ClothSeg
from extensions.CodeFormer.inference_codeformer_re import apply_codeformer
from extensions.HumanParsing.inference.inference_single import HumanParser
from extensions.midas import MidasDetector
from extensions.yolov8_opencv_dnn.main import YOLOv8_face

import insightface
from insightface.app import FaceAnalysis

# set init args
device = torch.device("cuda")
dtype = torch.float16
input_size = (1024, 1024)


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


def border_replicate(img_file):
    ori_w, ori_h = img_file.size
    pad_height = int(ori_h * 0.3)
    init_image = cv2.copyMakeBorder(np.array(img_file), pad_height, 0, 0, 0, cv2.BORDER_REPLICATE)
    return init_image


def pad_color(init_image, color):
    if color == 'white': value = (255, 255, 255)
    if color == 'black': value = (0, 0, 0)
    if init_image.size[0] < init_image.size[1]:
        bgImg: Image.Image = Image.new("RGB", (init_image.size[1], init_image.size[1]), value)
        bgImg.paste(init_image, (round((init_image.size[1] - init_image.size[0]) / 2), 0))
        init_image = bgImg.resize(input_size)
    else:
        bgImg: Image.Image = Image.new("RGB", (init_image.size[0], init_image.size[0]), value)
        bgImg.paste(init_image, (0, round((init_image.size[0] - init_image.size[1]) / 2)))
        init_image = bgImg.resize(input_size)
    return init_image


def pad_gaussian_noise(ori_img, new_img_h, new_img_w):
    # 生成高斯噪声
    bgImg = np.random.normal(0, 1, (new_img_h, new_img_w, 3))
    bgImg = (bgImg - bgImg.min()) / (bgImg.max() - bgImg.min()) * 255
    bgImg = Image.fromarray(np.uint8(bgImg))
    bgImg.paste(ori_img, (0, new_img_h - ori_img.size[1]))
    init_image = bgImg
    if init_image.size[0] < init_image.size[1]:
        bgImg: Image.Image = Image.new("RGB", (init_image.size[1], init_image.size[1]), (255, 255, 255))
        bgImg.paste(init_image, (round((init_image.size[1] - init_image.size[0]) / 2), 0))
        init_image = bgImg.resize(input_size)
    else:
        bgImg: Image.Image = Image.new("RGB", (init_image.size[0], init_image.size[0]), (255, 255, 255))
        bgImg.paste(init_image, (0, round((init_image.size[0] - init_image.size[1]) / 2)))
        init_image = bgImg.resize(input_size)
    return init_image


def random_gaussian_noise(img, mean, sigma, ori_mask_image):
    '''
    传入:
        img   :  原图
        mean  :  均值
        sigma :  标准差
    返回:
        gaussian_out : 噪声处理后的图片
        noise        : 对应的噪声
    '''
    img = img / 255
    noise = np.random.normal(mean, sigma, img.shape)
    gaussian_out = img.copy()
    gaussian_out[ori_mask_image < 127.5] = noise[ori_mask_image < 127.5]
    gaussian_out = np.uint8(gaussian_out * 255)
    return gaussian_out


# set init condition models
pose_path = f"models/ControlNet/control_v11p_sd15_openpose"
canny_path = f"models/ControlNet/control_v11p_sd15_canny"
depth_path = f"models/ControlNet/control_v11p_sd15_depth"
pose_model = ControlNetModel.from_pretrained(pose_path, torch_dtype=dtype).to(device)
canny_model = ControlNetModel.from_pretrained(canny_path, torch_dtype=dtype).to(device)
depth_model = ControlNetModel.from_pretrained(depth_path, torch_dtype=dtype).to(device)

# set other models
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
face_restored_path = "models/Extensions/CodeFormer/codeformer-v0.1.0.pth"

clotheseg_path = "/xsl/wilson.xu/diffusers/extensions/ClotheSeg/cloth_seg_v5.0.pt"
clotheseg_inferencer = ClothSeg(clotheseg_path, device=device)


class HeadInpainter():
    def __init__(self, mote_key, style_key):
        self.mote_key = mote_key
        self.scene_config = "photorealistic (clear background:1.4), highest quality settings, high quality, simple and clean"
        self.mote_config = {
            "asian_boy": {"base_path": "models/mote/braBeautifulRealistic_v3",
                          "prompt": "((RAW)), analog style, A stunning portrait of a man. he should smile and have  a captivating gaze, and an alluring expression. The background should be neutral and simple, allowing the man to be the image's focal point. Please pay close attention to the details, to create a lifelike and realistic image. The final product should be a masterpiece that captures the essence and beauty of the man, ((highly detailed skin, skin details)), sharp focus, volumetric fog, 8k UHD, DSLR, high quality, film grain, Fujifilm XT3 (black hair)",
                          "negative_prompt": "illumination, ng_deepnegative_v1_75t, paintings, sketches, (worst quality, low quality, normal quality:2), lowres, ((monochrome, grayscale)), skin spots, acnes, skin blemishes, age spot, backlight, ugly, Overexposure,(((tilted head))),((cut off))",
                          "seed": None},
            "asian_girl": {"base_path": "models/mote/braBeautifulRealistic_v3",
                           "prompt": "((RAW)), analog style, A stunning portrait of a young girl. Black eyes. She should smile and have a captivating gaze, and an alluring expression. The background should be neutral and simple, allowing the woman to be the image's focal point. Please pay close attention to the details, to create a lifelike and realistic image. The final product should be a masterpiece that captures the essence and beauty of the woman, ((highly detailed skin, skin details)), sharp focus, volumetric fog, 8k UHD, DSLR, high quality, film grain, Fujifilm XT3 (black hair)",
                           "negative_prompt": "illumination, ng_deepnegative_v1_75t, (badhandv4), (worst quality:2), (low quality:2), (normal quality:2), hat, lowres, bad anatomy, bad hands, ((monochrome)), ((grayscale)) watermark, (moles:2)",
                           "seed": None},
            "european_boy": {"base_path": "models/mote/CyberRealistic_V3",
                             "prompt": "white skin, ((photorealism)),(4k textures),natural skin texture, 24mm, 4k textures, soft cinematic light, RAW photo, photorealism, photorealistic, intricate, elegant, highly detailed, sharp focus, A stunning portrait of a young man. he should smile and have  a captivating gaze, and an alluring expression. The background should be neutral and simple, allowing the man to be the image's focal point. Please pay close attention to the details, to create a lifelike and realistic image.  ((masterpiece)), ((highly detailed skin, skin details)), sharp focus,",
                             "negative_prompt": "illumination, (deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), (depth of field:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck",
                             "seed": None},
            "european_girl": {"base_path": "models/mote/CyberRealistic_V3",
                              "prompt": "((RAW)), analog style, A stunning portrait of a young girl. Blue eyes. She should smlie and  have  a captivating gaze, and an alluring expression. The background should be neutral and simple, allowing the woman to be the image's focal point. Please pay close attention to the details, to create a lifelike and realistic image. The final product should be a masterpiece that captures the essence and beauty of the woman, ((highly detailed skin, skin details)), sharp focus",
                              "negative_prompt": "illumination, (deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck",
                              "seed": None},
        }
        self.style_config = {
            "european_girl":
                {"european_girl_sweet": {
                    "lora_model_path": ["models/mote/sweet0_eu_female0_1024x1024_10epoches.safetensors",
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
                 "european_girl_noble": {
                     "lora_model_path": ["models/mote/noble0_eu_female0_1024x1024_10epoches.safetensors",
                                         "models/mote/noble1_eu_female1_1024x1024_less_makeup_10epoches.safetensors"],
                     "ratio": [0.4, 0.4]}},  # 高贵

            "european_boy":
                {"european_boy_pretty": {
                    "lora_model_path": ["models/mote/handsome2_eu_male2_1024x1024_10epoches.safetensors",
                                        "models/mote/handsome4_eu_male4_1024x1024_10epoches.safetensors"],
                    "ratio": [0.2, 0.2]},  # 俊秀
                 "european_boy_new": {
                     "lora_model_path": ["models/mote/fashion0_eu_male0_1024x1024_10epoches.safetensors",
                                         "models/mote/fashion2_eu_male2_1024x1024_10epoches.safetensors"],
                     "ratio": [0.4, 0.4]},  # 新潮
                 "european_boy_elegance": {
                     "lora_model_path": ["models/mote/gentle0_eu_male0_1024x1024_10epoches.safetensors",
                                         "models/mote/gentle1_eu_male1_1024x1024_10epoches.safetensors"],
                     "ratio": [0.4, 0.4]},  # 儒雅
                 "european_boy_nature": {
                     "lora_model_path": ["models/mote/nature2_eu_male2_1024x1024_10epoches.safetensors",
                                         "models/mote/nature4_eu_male4_1024x1024_10epoches.safetensors"],
                     "ratio": [0.4, 0.4]},  # 自然
                 "european_boy_newandvig": {
                     "lora_model_path": ["models/mote/sharp0_eu_male0_1024x1024_10epoches.safetensors",
                                         "models/mote/sharp1_eu_male1_1024x1024_10epoches.safetensors"],
                     "ratio": [0.4, 0.4]},  # 新锐
                 "european_boy_romantic": {
                     "lora_model_path": ["models/mote/romantic0_eu_male0_1024x1024_10epoches.safetensors",
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
                 "asian_girl_noble": {
                     "lora_model_path": ["models/mote/noble0_as_female0_1024x1024_10epoches.safetensors",
                                         "models/mote/noble1_as_female1_1024x1024_10epoches.safetensors"],
                     "ratio": [0.4, 0.4]}},  # 高贵     # 随机

            "asian_boy":
                {"asian_boy_pretty": {
                    "lora_model_path": ["models/mote/handsome2_as_male2_1024x1024_10epoches.safetensors",
                                        "models/mote/handsome3_as_male3_1024x1024_10epoches.safetensors"],
                    "ratio": [0.4, 0.4]},  # 俊秀
                 "asian_boy_new": {"lora_model_path": ["models/mote/fashion0_as_male0_1024x1024_10epoches.safetensors",
                                                       "models/mote/fashion1_as_male1_1024x1024_10epoches.safetensors"],
                                   "ratio": [0.4, 0.4]},  # 新潮
                 "asian_boy_elegance": {
                     "lora_model_path": ["models/mote/gentle0_as_male0_1024x1024_10epoches.safetensors",
                                         "models/mote/gentle2_as_male2_1024x1024_10epoches.safetensors"],
                     "ratio": [0.4, 0.4]},  # 儒雅
                 "asian_boy_nature": {
                     "lora_model_path": ["models/mote/nature0_as_male0_1024x1024_10epoches.safetensors",
                                         "models/mote/nature1_as_male1_1024x1024_10epoches.safetensors"],
                     "ratio": [0.4, 0.4]},  # 自然
                 "asian_boy_newandvig": {
                     "lora_model_path": ["models/mote/sharp0_as_male0_1024x1024_10epoches.safetensors",
                                         "models/mote/sharp1_as_male1_1024x1024_10epoches.safetensors"],
                     "ratio": [0.2, 0.6]},  # 新锐
                 "asian_boy_romantic": {
                     "lora_model_path": ["models/mote/romantic0_as_male0_1024x1024_10epoches.safetensors",
                                         "models/mote/romantic1_as_male1_1024x1024_10epoches.safetensors"],
                     "ratio": [0.6, 0.2]},  # 浪漫
                 "asian_boy_arrogance": {
                     "lora_model_path": ["models/mote/domineer0_as_male0_1024x1024_10epoches.safetensors",
                                         "models/mote/domineer1_as_male1_1024x1024_10epoches.safetensors"],
                     "ratio": [0.4, 0.4]}},  # 霸气
        }
        self.prompt = f"{self.mote_config[mote_key]['prompt']}, {self.scene_config}"
        self.negative_prompt = self.mote_config[mote_key]['negative_prompt']
        base_path = self.mote_config[mote_key]["base_path"]
        controlnet = [pose_model, canny_model]
        self.pipe_control = VIPStableDiffusionControlNetInpaintPipeline.from_pretrained(base_path,
                                                                                        controlnet=controlnet,
                                                                                        torch_dtype=dtype).to(device)
        if "random" in style_key:
            style_key = random.choice(list(self.style_config[mote_key]))

        # for i in range(len(self.style_config[mote_key][style_key]["lora_model_path"])):
        #     lora_model_path = self.style_config[mote_key][style_key]["lora_model_path"][i]
        #     lora_ratio = self.style_config[mote_key][style_key]["ratio"][i]
        #     self.pipe_control = load_lora_weights(self.pipe_control, lora_model_path, lora_ratio, device="cuda",
        #                                           dtype=dtype)
        # self.pipe_control.load_lora_weights("sd-model-finetuned-lora", weight_name="pytorch_lora_weights.safetensors")
        # self.pipe_control.fuse_lora(lora_scale=1.0)
        self.pipe_control.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe_control.scheduler.config)
        load_webui_textual_inversion("embedding", self.pipe_control)

        self.get_depth = MidasDetector()
        # self.clothe_seg_model = SegModel('extensions/ClotheSeg/', input_size[0])
        self.HumanParse = HumanParser(
            model_path="models/Extensions/2D-Human-Parsing/deeplabv3plus-xception-vocNov14_20-51-38_epoch-89.pth")

    def __call__(self, mote, batch_size, key):
        self.seed = get_fixed_seed(self.mote_config[self.mote_key]["seed"])
        print(f'processing seed = {self.seed}')
        if key == 1:
            ori_w, ori_h = mote.size
            # 判断头部是否贴上边界
            label_parsing = self.HumanParse.inference(mote).astype('uint8')
            # label = {0:"background", 1:"hat", 2:"hair", 4:"sunglasses", 10: "neck", 13:"face"}
            face_mask = np.zeros_like(label_parsing)
            face_mask[label_parsing == 2] = 255
            face_mask[label_parsing == 10] = 255
            face_mask[label_parsing == 13] = 255
            y1 = np.where(np.sum(face_mask, axis=1))[0][0]
            if y1 > 0:
                y1 += 10
            mote = np.array(mote)[y1:]
            h = mote.shape[0]
            fill_img = np.full([int(h * 1.3), ori_w, 3], 255, dtype=np.uint8)
            fill_img[fill_img.shape[0] - h:] = mote

            pose_image, dic = pose_inferencer(fill_img)
            pad_height = int(
                (dic['body']['keypoints'][0][1][1] - dic['body']['keypoints'][0][0][1]) * fill_img.shape[0] * 1.3)

            mote = cv2.copyMakeBorder(np.array(mote), pad_height, 0, 0, 0, cv2.BORDER_REPLICATE)
            distance = mote.shape[0] - pose_image.shape[0]
            if distance < 0:
                pose_image = pose_image[abs(distance):, :]
            else:
                pose_image = cv2.copyMakeBorder(pose_image, abs(distance), 0, 0, 0, cv2.BORDER_CONSTANT,
                                                value=(0, 0, 0))
            pose_image = pad_color(Image.fromarray(pose_image), 'black')

            mote = pad_color(Image.fromarray(mote), 'white')
            init_image = load_image(mote)

            strength = 0.8
            mask = np.zeros((input_size[0], input_size[1]), np.uint8)
            mask[:int(pad_height / (h + pad_height) * 1024)] = 255
            # label_parsing = self.HumanParse.inference(mote).astype('uint8')
            # mask[label_parsing == 2] = 255
            # mask[label_parsing == 10] = 255
            # mask[label_parsing == 13] = 255
            # kernel = np.ones([9, 9], np.uint8)
            # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask_image = mask_process(Image.fromarray(mask), invert_mask=False, blur=4)
        else:
            init_image = load_image(mote)

            strength = 0.5
            label_parsing = self.HumanParse.inference(init_image).astype('uint8')
            head_mask = np.zeros_like(label_parsing)
            # label = {0:"background", 1:"hat", 2:"hair", 4:"sunglasses", 10: "neck", 13:"face"}
            for label in [1, 2, 4, 10, 13]:
                head_mask[label_parsing == label] = 255
            # kernel = np.ones((5, 5), dtype=np.uint8)
            # head_mask = cv2.dilate(head_mask, kernel, 1)
            mask_image = mask_process(Image.fromarray(head_mask), invert_mask=False)

            pose_image, dic = pose_inferencer(init_image)
            pose_image = Image.fromarray(pose_image)

        generator = get_torch_generator(self.seed, batch_size, device=device)

        clo_mask = np.array(clotheseg_inferencer(init_image))
        canny_image = np.array(init_image)
        canny_image[clo_mask != 255] = 0
        canny_image = cv2.Canny(canny_image, 100, 200)
        canny_image = np.tile(canny_image[..., None], [1, 1, 3]).astype('uint8')
        canny_image = Image.fromarray(canny_image)

        pipe_list, image_overlay = self.pipe_control(
            prompt=self.prompt,
            negative_prompt=self.negative_prompt,
            image=init_image,
            mask_image=mask_image,
            control_image=[pose_image, canny_image],
            height=input_size[1],
            width=input_size[0],
            strength=strength,
            num_inference_steps=20,
            guidance_scale=7,
            num_images_per_prompt=batch_size,
            generator=generator,
            controlnet_conditioning_scale=[1., 1.])
        pipe_list = apply_codeformer(face_restored_path, pipe_list)
        pipe_list = alpha_composite(pipe_list, image_overlay)
        return pipe_list


# 同ID换脸
class Adetailer():
    def __init__(self, mote_key, style_key):
        self.mote_key = mote_key
        self.mote_config = {
            "asian_boy": {"base_path": "models/mote/braBeautifulRealistic_v3",
                          "prompt": "((RAW)), analog style, A stunning portrait of a man. he should smile and have  a captivating gaze, and an alluring expression. The background should be neutral and simple, allowing the man to be the image's focal point. Please pay close attention to the details, to create a lifelike and realistic image. The final product should be a masterpiece that captures the essence and beauty of the man, ((highly detailed skin, skin details)), sharp focus, volumetric fog, 8k UHD, DSLR, high quality, film grain, Fujifilm XT3 (black hair)",
                          "negative_prompt": "ng_deepnegative_v1_75t, paintings, sketches, (worst quality, low quality, normal quality:2), lowres, ((monochrome, grayscale)), skin spots, acnes, skin blemishes, age spot, backlight, ugly, Overexposure,(((tilted head))),((cut off))",
                          "seed": None},
            "asian_girl": {"base_path": "models/mote/braBeautifulRealistic_v3",
                           "prompt": "((RAW)), analog style, A stunning portrait of a young girl. Black eyes. She should smile and have a captivating gaze, and an alluring expression. The background should be neutral and simple, allowing the woman to be the image's focal point. Please pay close attention to the details, to create a lifelike and realistic image. The final product should be a masterpiece that captures the essence and beauty of the woman, ((highly detailed skin, skin details)), sharp focus, volumetric fog, 8k UHD, DSLR, high quality, film grain, Fujifilm XT3 (black hair)",
                           "negative_prompt": "ng_deepnegative_v1_75t, (badhandv4), (worst quality:2), (low quality:2), (normal quality:2), hat, lowres, bad anatomy, bad hands, ((monochrome)), ((grayscale)) watermark, (moles:2)",
                           "seed": None},
            "european_boy": {"base_path": "models/mote/CyberRealistic_V3",
                             "prompt": "white skin, ((photorealism)),(4k textures),natural skin texture, 24mm, 4k textures, soft cinematic light, RAW photo, photorealism, photorealistic, intricate, elegant, highly detailed, sharp focus, A stunning portrait of a young man. he should smile and have  a captivating gaze, and an alluring expression. The background should be neutral and simple, allowing the man to be the image's focal point. Please pay close attention to the details, to create a lifelike and realistic image.  ((masterpiece)), ((highly detailed skin, skin details)), sharp focus,",
                             "negative_prompt": "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), (depth of field:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck",
                             "seed": None},
            "european_girl": {"base_path": "models/mote/CyberRealistic_V3",
                              "prompt": "((RAW)), analog style, A stunning portrait of a young girl. Blue eyes. She should smlie and  have  a captivating gaze, and an alluring expression. The background should be neutral and simple, allowing the woman to be the image's focal point. Please pay close attention to the details, to create a lifelike and realistic image. The final product should be a masterpiece that captures the essence and beauty of the woman, ((highly detailed skin, skin details)), sharp focus",
                              "negative_prompt": "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck",
                              "seed": None},
        }
        self.style_config = {
            "european_girl":
                {"european_girl_sweet": {
                    "lora_model_path": ["models/mote/sweet0_eu_female0_1024x1024_10epoches.safetensors",
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
                 "european_girl_noble": {
                     "lora_model_path": ["models/mote/noble0_eu_female0_1024x1024_10epoches.safetensors",
                                         "models/mote/noble1_eu_female1_1024x1024_less_makeup_10epoches.safetensors"],
                     "ratio": [0.4, 0.4]}},  # 高贵

            "european_boy":
                {"european_boy_pretty": {
                    "lora_model_path": ["models/mote/handsome2_eu_male2_1024x1024_10epoches.safetensors",
                                        "models/mote/handsome4_eu_male4_1024x1024_10epoches.safetensors"],
                    "ratio": [0.2, 0.2]},  # 俊秀
                 "european_boy_new": {
                     "lora_model_path": ["models/mote/fashion0_eu_male0_1024x1024_10epoches.safetensors",
                                         "models/mote/fashion2_eu_male2_1024x1024_10epoches.safetensors"],
                     "ratio": [0.4, 0.4]},  # 新潮
                 "european_boy_elegance": {
                     "lora_model_path": ["models/mote/gentle0_eu_male0_1024x1024_10epoches.safetensors",
                                         "models/mote/gentle1_eu_male1_1024x1024_10epoches.safetensors"],
                     "ratio": [0.4, 0.4]},  # 儒雅
                 "european_boy_nature": {
                     "lora_model_path": ["models/mote/nature2_eu_male2_1024x1024_10epoches.safetensors",
                                         "models/mote/nature4_eu_male4_1024x1024_10epoches.safetensors"],
                     "ratio": [0.4, 0.4]},  # 自然
                 "european_boy_newandvig": {
                     "lora_model_path": ["models/mote/sharp0_eu_male0_1024x1024_10epoches.safetensors",
                                         "models/mote/sharp1_eu_male1_1024x1024_10epoches.safetensors"],
                     "ratio": [0.4, 0.4]},  # 新锐
                 "european_boy_romantic": {
                     "lora_model_path": ["models/mote/romantic0_eu_male0_1024x1024_10epoches.safetensors",
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
                 "asian_girl_noble": {
                     "lora_model_path": ["models/mote/noble0_as_female0_1024x1024_10epoches.safetensors",
                                         "models/mote/noble1_as_female1_1024x1024_10epoches.safetensors"],
                     "ratio": [0.4, 0.4]}},  # 高贵     # 随机

            "asian_boy":
                {"asian_boy_pretty": {
                    "lora_model_path": ["models/mote/handsome2_as_male2_1024x1024_10epoches.safetensors",
                                        "models/mote/handsome3_as_male3_1024x1024_10epoches.safetensors"],
                    "ratio": [0.4, 0.4]},  # 俊秀
                 "asian_boy_new": {"lora_model_path": ["models/mote/fashion0_as_male0_1024x1024_10epoches.safetensors",
                                                       "models/mote/fashion1_as_male1_1024x1024_10epoches.safetensors"],
                                   "ratio": [0.4, 0.4]},  # 新潮
                 "asian_boy_elegance": {
                     "lora_model_path": ["models/mote/gentle0_as_male0_1024x1024_10epoches.safetensors",
                                         "models/mote/gentle2_as_male2_1024x1024_10epoches.safetensors"],
                     "ratio": [0.4, 0.4]},  # 儒雅
                 "asian_boy_nature": {
                     "lora_model_path": ["models/mote/nature0_as_male0_1024x1024_10epoches.safetensors",
                                         "models/mote/nature1_as_male1_1024x1024_10epoches.safetensors"],
                     "ratio": [0.4, 0.4]},  # 自然
                 "asian_boy_newandvig": {
                     "lora_model_path": ["models/mote/sharp0_as_male0_1024x1024_10epoches.safetensors",
                                         "models/mote/sharp1_as_male1_1024x1024_10epoches.safetensors"],
                     "ratio": [0.2, 0.6]},  # 新锐
                 "asian_boy_romantic": {
                     "lora_model_path": ["models/mote/romantic0_as_male0_1024x1024_10epoches.safetensors",
                                         "models/mote/romantic1_as_male1_1024x1024_10epoches.safetensors"],
                     "ratio": [0.6, 0.2]},  # 浪漫
                 "asian_boy_arrogance": {
                     "lora_model_path": ["models/mote/domineer0_as_male0_1024x1024_10epoches.safetensors",
                                         "models/mote/domineer1_as_male1_1024x1024_10epoches.safetensors"],
                     "ratio": [0.4, 0.4]}},  # 霸气
        }
        self.prompt = f"{self.mote_config[mote_key]['prompt']}"
        self.negative_prompt = self.mote_config[mote_key]['negative_prompt']
        base_path = self.mote_config[mote_key]["base_path"]
        controlnet = [canny_model]
        self.pipe_control = VIPStableDiffusionControlNetInpaintPipeline.from_pretrained(base_path,
                                                                                        controlnet=controlnet,
                                                                                        torch_dtype=dtype).to(device)
        if "random" in style_key:
            style_key = random.choice(list(self.style_config[mote_key]))
        # for i in range(len(self.style_config[mote_key][style_key]["lora_model_path"])):
        #     lora_model_path = self.style_config[mote_key][style_key]["lora_model_path"][i]
        #     lora_ratio = self.style_config[mote_key][style_key]["ratio"][i]
        #     self.pipe_control = load_lora_weights(self.pipe_control, lora_model_path, lora_ratio, device="cuda",
        #                                           dtype=dtype)
        # self.pipe_control.load_lora_weights("sd-model-finetuned-lora", weight_name="pytorch_lora_weights.safetensors")
        self.pipe_control.load_lora_weights("/xsl/wilson.xu/lora_face", weight_name="progress1_1.safetensors")
        self.pipe_control.fuse_lora(lora_scale=1.0)
        self.pipe_control.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe_control.scheduler.config)
        load_webui_textual_inversion("embedding", self.pipe_control)

        # self.HumanParse = HumanParser(model_path="models/Extensions/2D-Human-Parsing/deeplabv3plus-xception-vocNov14_20-51-38_epoch-89.pth")
        self.YOLOv8_face_detector = YOLOv8_face("extensions/yolov8_opencv_dnn/weights/yolov8n-face.onnx",
                                                conf_thres=0.45, iou_thres=0.5)

        self.seed = get_fixed_seed(self.mote_config[self.mote_key]["seed"])
        print(f'processing seed = {self.seed}')

    def __call__(self, mote, batch_size):
        init_image = load_image(mote)

        motemask = postApi_zuotang(init_image).convert("L")
        motemask = np.array(motemask)

        canny_image = np.array(init_image).copy()
        canny_image[motemask < 127] = 0
        canny_image = cv2.Canny(canny_image, 100, 200)[..., None]
        canny_image = Image.fromarray(np.tile(canny_image, [1, 1, 3]))

        bboxes, masks, preview = mediapipe_face_detection(init_image)
        if masks is None:
            boxes, scores, classids, kpts = self.YOLOv8_face_detector.detect(np.array(init_image))
            if len(boxes) == 0:
                return [init_image] * batch_size
            mask = np.zeros((input_size[0], input_size[1], 3), np.uint8)
            x, y, w, h = boxes[0].astype(int)
            if w > h:
                mask[y - (w - h) // 2:y + w - (w - h) // 2, x:x + w] = 255
            else:
                mask[y:y + h, x - (h - w) // 2:x + h - (h - w) // 2] = 255
            mask = mask_process(Image.fromarray(mask), invert_mask=False)
        else:
            mask = mask_process(masks[0], invert_mask=False)

        crop_region = get_crop_region(np.array(mask), pad=32)
        crop_region = expand_crop_region(crop_region, input_size[1], input_size[0], mask.width, mask.height)
        mask = mask.crop(crop_region)
        img = init_image.crop(crop_region)

        generator = get_torch_generator(self.seed, batch_size, device=device)
        pipe_list, image_overlay = self.pipe_control(
            prompt=self.prompt,
            negative_prompt=self.negative_prompt,
            image=img,
            mask_image=mask,
            control_image=[canny_image],
            height=512,
            width=512,
            strength=0.2,
            num_inference_steps=30,
            guidance_scale=6,
            num_images_per_prompt=batch_size,
            generator=generator,
            controlnet_conditioning_scale=0.,
            enhance='sharpness',
            )  # 控制是否使用canny
        # pipe_list = apply_codeformer(face_restored_path, pipe_list)
        pipe_list = alpha_composite(pipe_list, image_overlay)
        x1, y1, x2, y2 = crop_region
        final = list()
        for i, out_img in enumerate(pipe_list):
            res_image = init_image.copy()
            res_image.paste(out_img.resize((int(x2 - x1), int(y2 - y1)), resample=Image.LANCZOS), (x1, y1))
            final.append(res_image)
        return final


if __name__ == "__main__":
    import time, glob

    start_time = time.time()

    init_image_root = f"/xsl/wilson.xu/half_face"
    save_dir = "output"
    os.makedirs(save_dir, exist_ok=True)

    gentle = "girl"
    style = "vanguard"
    country = "asian"
    head_inpainter = HeadInpainter(mote_key=f"{country}_{gentle}", style_key=f"{country}_{gentle}_{style}")
    face_inpainter = Adetailer(mote_key=f"{country}_{gentle}", style_key=f"{country}_{gentle}_{style}")

    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(640, 640))
    swapper = insightface.model_zoo.get_model('/root/.insightface/models/inswapper_128.onnx')

    # dst_img = cv2.imread('/xsl/wilson.xu/dataset/face_images/#10_progress0_as_female0_1024x1024/1.jpg')
    # dst_img = cv2.imread('/xsl/wilson.xu/dataset/face_images/#10_progress1_as_female1_1024x1024/17.jpg')
    dst_img = cv2.imread('/xsl/wilson.xu/dataset/face_images/#10_progress1_as_female1_1024x1024/17.jpg')
    dst_face = app.get(dst_img)[0]

    print(f"init cost {time.time() - start_time}")
    start_time = time.time()

    # global name
    img_list = os.listdir(init_image_root)
    for name in tqdm(img_list):
        if os.path.splitext(name)[-1].lower() not in ['.png', '.jpg', '.jpeg']:
            continue

        init_img = Image.open(os.path.join(init_image_root, name))
        head_res = head_inpainter(mote=init_img, batch_size=1, key=1)[0]  # key=1即第一次执行
        head_res = head_inpainter(mote=head_res, batch_size=1, key=2)[0]  # key=1即第一次执行

        print(f"1st inpaint cost {time.time() - start_time}")
        start_time = time.time()

        src_img = np.array(head_res)[..., ::-1]
        src_face = app.get(src_img)
        res = swapper.get(src_img.copy(), src_face[0], dst_face, paste_back=True)
        res = Image.fromarray(res[..., ::-1])

        face_res = face_inpainter(mote=res, batch_size=3)

        print(f"Adetailer inpaint cost {time.time() - start_time}")
        final_res = np.concatenate([init_img.resize(input_size), head_res, res], axis=1)
        for i, res in enumerate(face_res):
            final_res = np.concatenate([final_res, res], axis=1)
        Image.fromarray(final_res).save(
                f"{save_dir}/{country}_{gentle}_{style}_{name.replace('.png', '').replace('.jpg', '')}.jpg")
