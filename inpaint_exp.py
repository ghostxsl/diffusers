import os
import numpy as np
import cv2
from PIL import Image
from diffusers import ControlNetModel
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from diffusers.schedulers import EulerAncestralDiscreteScheduler, KDPMPP2MDiscreteScheduler
from diffusers import VIPStableDiffusionControlNetInpaintPipeline, \
    UniPCMultistepScheduler

from extensions.CodeFormer.inference_codeformer_re import apply_codeformer
from mmpose.blendpose.openpose import OpenposeDetector
from mmpose.blendpose import VIPPoseInferencer
from diffusers.utils.promt_parser import load_webui_textual_inversion
from diffusers.utils.vip_utils import *


def get_controlnet(pro_type, root_dir, dtype=torch.float16, device=torch.device("cuda")):
    if pro_type[0] == 'a':
        base_path = f'{root_dir}/weights/braBeautifulRealistic_brav3'
    else:
        base_path = f'{root_dir}/weights/CyberRealistic_V3.0-FP32'

    pose_path = f'{root_dir}/weights/exp_control_v11p_sd15_openpose'
    canny_path = f'{root_dir}/weights/control_v11p_sd15_canny'
    controlnet1 = ControlNetModel.from_pretrained(pose_path, torch_dtype=dtype).to(device)
    controlnet2 = ControlNetModel.from_pretrained(canny_path, torch_dtype=dtype).to(device)
    controlnet = MultiControlNetModel([controlnet1, controlnet2])

    pipe_control = VIPStableDiffusionControlNetInpaintPipeline.from_pretrained(
        base_path, controlnet=controlnet, torch_dtype=dtype).to(device)
    pipe_control.scheduler = KDPMPP2MDiscreteScheduler.from_config(pipe_control.scheduler.config)
    load_webui_textual_inversion("embedding", pipe_control)
    pipe_control.enable_model_cpu_offload()
    # pipe_control.enable_xformers_memory_efficient_attention()
    return pipe_control


pro_types = ["asian_woman", "asian_man", "european_woman", "european_man"]
pro_type = pro_types[0]
prompt_dict = {
    "asian_man": "1 asian boy, photo of a full body shot of a handsome young man, smile, abs, clear grey background:1.4, best quality, masterpiece, ultra highres,  very detailed skin, 4k, photorealistic, masterpiece, very high detailed",
    "asian_woman": "1 girl, photo of a full body shot of a beautiful woman, black hair, black eyes, smile, abs, clear grey background:1.4, best quality, masterpiece, ultra highres,  very detailed skin, 4k, photorealistic, masterpiece, very high detailed",
    "european_man": "1 European young man, photo of a full body shot of a handsome young man, smile, necklace,abs, clear grey background:1.4, best quality, masterpiece, ultra highres,  very detailed skin, 4k, photorealistic, masterpiece, very high detailed",
    "european_woman": "1 European girl, photo of a full body shot of a beautiful woman, smile, Blond hair, abs, shoes, blue eyes, clear grey background:1.4, best quality, masterpiece, ultra highres,  very detailed skin, 4k, photorealistic, masterpiece, very high detailed"
}

init_root_dir = "/xsl/wilson.xu"
save_mask_root = f"{init_root_dir}/mannequin_0620/mask"
init_image_root = f"{init_root_dir}/mannequin_0620"
face_restored_path = f"{init_root_dir}/weights/codeformer-v0.1.0.pth"

dtype = torch.float16
device = torch.device("cuda")

input_size = (1024, 1024)
seed = get_fixed_seed(1385966638)
batch_size = 1
generator = get_torch_generator(seed, batch_size, device=device)
print(f'processing seed = {seed}')

save_dir = f'./output'
os.makedirs(save_dir, exist_ok=True)

pipe_control = get_controlnet(pro_type, init_root_dir, dtype, device)

det_config = "/xsl/wilson.xu/mmpose/mmpose/blendpose/configs/rtmdet_l_8xb32-300e_coco.py"
det_checkpoint = "/xsl/wilson.xu/weights/rtmpose_model/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth"
body_config = "/xsl/wilson.xu/mmpose/configs/body_2d_keypoint/rtmpose/body8/rtmpose-l_8xb256-420e_body8-256x192.py"
body_checkpoint = "/xsl/wilson.xu/weights/rtmpose_model/rtmpose-l_simcc-body7_pt-body7_420e-256x192-4dba18fc_20230504.pth"
wholebody_cfg = f"{init_root_dir}/mmpose/mmpose/blendpose/configs/dwpose_l_wholebody_384x288.py"
wholebody_pth = f"{init_root_dir}/weights/rtmpose_model/dw-ll_ucoco_384.pth"
pose_inferencer = VIPPoseInferencer(det_config, det_checkpoint,
                                    body_config, body_checkpoint,
                                    wholebodypose_cfg=wholebody_cfg,
                                    wholebodypose_pth=wholebody_pth,
                                    return_results=True,
                                    body_kpt_thr=0.3,
                                    hand_kpt_thr=0.3,
                                    device=device)

# pose_inferencer = OpenposeDetector(use_hand=True,
#                                    body_pth="/xsl/wilson.xu/weights/openpose_model/body_pose_model.pth",
#                                    hand_pth="/xsl/wilson.xu/weights/openpose_model/hand_pose_model.pth",
#                                    return_results=True,
#                                    device=device)

names = os.listdir(os.path.join(save_mask_root, pro_type.split('_')[1]))
for i, name in enumerate(names):
    # name = "IMG_9817"
    img_path = os.path.join(init_image_root, pro_type.split('_')[1], name + '.JPG')
    print(f"process {img_path}")
    init_image = load_image(img_path).resize(input_size, resample=Image.LANCZOS)
    print(f"image_size = {init_image.size}")

    clo_mask_path = os.path.join(save_mask_root, pro_type.split('_')[1], name, "clothing_mask.png")
    ori_mask = load_image(clo_mask_path)
    clo_mask = ori_mask.resize(input_size, resample=Image.LANCZOS)

    prompt = prompt_dict[pro_type] + ", no other attachments"
    bboxes, masks, preview = mediapipe_face_detection(init_image)
    if bboxes is None:
        prompt = "photo of a person with back, " + prompt
        print("back!")
    negative_prompt = "ng_deepnegative_v1_75t, high light on body, (badhandv4), (worst quality:2), (low quality:2), (normal quality:2), lowres, ((monochrome)), ((grayscale)), glans, extra fingers, fewer fingers, (multi nipples), bad anatomy, bad hands, text, error, missing fingers, missing arms, missing legs, extra digit, fewer digits, cropped, worst quality, jpeg artifacts, signature, watermark, username, bad feet, Multiple people, blurry, poorly drawn hands, poorly drawn face, mutation, deformed, extra limbs, extra arms, extra legs, malformed limbs, fused fingers, too many fingers, long neck, ((cross-eyed)), mutated hands, polar lowres, bad body, bad proportions, gross proportions, wrong feet bottom render, abdominal stretch, briefs, knickers, kecks, thong, fused fingers, bad body,bad proportion body to legs, wrong toes, extra toes, missing toes, weird toes, 2 body, 2 pussy, 2 upper, 2 lower, 2 head, 3 hand, 3 feet, extra long leg, super long leg, mirrored image, mirrored noise, skin spots, acnes, age spot, ((watermark:2)), (white letters:1), illustration, 3d, sepia, painting, cartoon, sketch"

    # get pose
    pose_image, pose_res = pose_inferencer(np.array(init_image), hand=True)
    pose_image = Image.fromarray(pose_image)
    # get canny
    canny_image = cv2.Canny(np.array(clo_mask), 100, 200)[..., None]
    canny_image = np.tile(canny_image, [1, 1, 3])
    canny_image = Image.fromarray(canny_image)

    # get inpaint mask
    mask_image = mask_process(ori_mask, invert_mask=True).resize(input_size, resample=Image.LANCZOS)

    # 1. Inpaint
    pipe1_list, image1_overlay = pipe_control(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=init_image,
        mask_image=mask_image,
        control_image=[pose_image, canny_image],
        height=input_size[0],
        width=input_size[1],
        strength=0.8,
        num_inference_steps=30,
        guidance_scale=9,  # CFGscale
        num_images_per_prompt=batch_size,
        generator=generator,
        controlnet_conditioning_scale=1.0, )

    pipe1_list = apply_codeformer(face_restored_path, pipe1_list)
    pipe1_list = alpha_composite(pipe1_list, image1_overlay)

    for i, img in enumerate(pipe1_list):
        # final = np.concatenate((final, img), axis=1).astype('uint8')
        # Image.fromarray(final).save(f"{save_dir}/{name}_{i}.png")
        img.save(f"{save_dir}/{name}_1_{i}.png")
        # pose_image = Image.fromarray(pose_inferencer(np.array(img), hand=True))

    # 2. Hand repair
    mask_pad = 32
    for j, img in enumerate(pipe1_list):
        # get pose
        pose_image, pose_res = pose_inferencer(np.array(img), hand=True)
        pose_image = Image.fromarray(pose_image)
        hand_kpts = pose_res['hand']['keypoints'][..., :2] * np.array(input_size)
        hand_scores = pose_res['hand']['keypoints'][..., 2]

        hand_bboxes = []
        for kpts, score in zip(hand_kpts, hand_scores):
            kpt = np.take(kpts, np.nonzero(score > 0.3)[0], axis=0)
            x1, y1 = np.min(kpt, axis=0)
            x2, y2 = np.max(kpt, axis=0)
            hand_bboxes.append([x1 - mask_pad, y1 - mask_pad, x2 + mask_pad, y2 + mask_pad])

        masks = create_mask_from_bbox(hand_bboxes, input_size)
        for mask in masks:
            mask = mask_process(mask, invert_mask=False)
            crop_region = get_crop_region(np.array(mask), pad=mask_pad)
            crop_region = expand_crop_region(crop_region,
                                             input_size[1], input_size[0],
                                             mask.width, mask.height)
            mask = mask.crop(crop_region)
            img2 = img.crop(crop_region)
            pose_image2 = pose_image.crop(crop_region)

            pipe2_list, image2_overlay = pipe_control(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=img2,
                mask_image=mask,
                control_image=[pose_image2, canny_image],
                height=input_size[0],
                width=input_size[1],
                strength=0.3,
                num_inference_steps=30,
                guidance_scale=9,  # CFGscale
                num_images_per_prompt=batch_size,
                generator=generator,
                controlnet_conditioning_scale=[1.0, 0.0], )
            pipe2_list = alpha_composite(pipe2_list, image2_overlay)
            # Paste back according to the mask
            x1, y1, x2, y2 = crop_region
            img.paste(
                pipe2_list[0].resize(
                    (int(x2 - x1), int(y2 - y1)), resample=Image.LANCZOS),
                (x1, y1))
        pipe1_list[j] = img
        pipe1_list[j].save(f"{save_dir}/{name}_2_{i}.png")

    # # 3. Hand repair
    # mask_pad = 32
    # for img in pipe1_list:
    #     # get pose
    #     pose_image, pose_res = pose_inferencer(np.array(img), hand=True)
    #     pose_image = Image.fromarray(pose_image)
    #     hand_kpts = pose_res['hand']['keypoints'][..., :2] * np.array(input_size)
    #     hand_scores = pose_res['hand']['keypoints'][..., 2]

    #     hand_bboxes = []
    #     for kpts, score in zip(hand_kpts, hand_scores):
    #         kpt = np.take(kpts, np.nonzero(score > 0.3)[0], axis=0)
    #         x1, y1 = np.min(kpt, axis=0)
    #         x2, y2 = np.max(kpt, axis=0)
    #         hand_bboxes.append([x1 - mask_pad, y1 - mask_pad, x2 + mask_pad, y2 + mask_pad])
    #     mask_image = create_mask_from_bbox_to_one_img(hand_bboxes, input_size)
    #     mask_image = mask_process(mask_image, invert_mask=False)

    #     pipe2_list, image2_overlay = pipe_control(
    #         prompt=prompt,
    #         negative_prompt=negative_prompt,
    #         image=img,
    #         mask_image=mask_image,
    #         control_image=[pose_image, canny_image],
    #         height=input_size[0],
    #         width=input_size[1],
    #         strength=0.4,
    #         num_inference_steps=30,
    #         guidance_scale=9,  # CFGscale
    #         num_images_per_prompt=batch_size,
    #         generator=generator,
    #         controlnet_conditioning_scale=0.0, )

    #     pipe2_list = alpha_composite(pipe2_list, image2_overlay)

    #     pipe2_list[0].save(f"{save_dir}/{name}_2_{i}.png")

    # final = np.concatenate([init_image.resize(input_size),
    #                         mask_image.convert("RGB").resize(input_size),
    #                         pose_image.resize(input_size),
    #                         canny_image.resize(input_size)], axis=1)
