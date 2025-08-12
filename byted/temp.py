import os
from os.path import join, splitext, basename
from tqdm import tqdm
import csv
import time
import json
import numpy as np
from PIL import Image

from diffusers.data.utils import load_file, csv_save, load_csv_or_xlsx_to_dict, xlsx_save, json_save
from diffusers.data.outer_vos_tools import encode_pil_bytes, decode_pil_bytes, load_or_download_image
from diffusers.data.tos import url_to_bytes, save_file_to_tos, save_tos, _gen_name




# # outpainting
# from diffusers.data.clients.toutiao_labcv_algo_vproxy import img2img_outpainting
# token = os.popen("doas -p ad.creative.image_core_solution printenv SEC_TOKEN_STRING|tail -1 ").read()
# os.environ["SEC_TOKEN_STRING"] = token
# if len(os.getenv("SEC_TOKEN_STRING", "")) < 1:
#     token = os.popen("cat /tmp/identity.token").read()
#     os.environ["SEC_TOKEN_STRING"] = token
# os.environ["SEC_KV_AUTH"] = "1"
# os.environ["TCE_PSM"] = "ad.creative.image_core_solution"
#
#
# def pad_image(image, padding, fill_color=(0, 0, 0)):
#     """
#     为图像添加填充
#
#     参数:
#         image: PIL Image对象
#         padding: 填充大小，可以是一个整数(四周填充相同大小)或元组(左,右,上,下)
#         fill_color: 填充区域的颜色，默认为白色(255,255,255)
#
#     返回:
#         填充后的PIL Image对象
#     """
#     # 处理填充参数
#     if isinstance(padding, int):
#         left = right = top = bottom = padding
#     else:
#         left, right, top, bottom = padding
#
#     # 获取原图尺寸
#     width, height = image.size
#
#     # 计算新图像尺寸
#     new_width = width + left + right
#     new_height = height + top + bottom
#
#     # 创建新图像，使用指定颜色填充
#     new_image = Image.new(image.mode, (new_width, new_height), fill_color)
#
#     # 将原图粘贴到新图像的指定位置
#     new_image.paste(image, (left, top))
#
#     return new_image
#
#
# row_dict = load_csv_or_xlsx_to_dict("/mnt/bn/ttcc-algo-bytenas/zjn/data/poster_train/results_all_done_4567.csv")
#
# width, height = 720, 1280
# for line in tqdm(row_dict):
#     img = load_or_download_image(line['gen_url'])
#     w, h = img.size
#     oh = int(h * width / w)
#     img = img.resize((width, oh), 1)
#     pad_ = (height - oh) // 2
#     img = pad_image(img, (0, 0, pad_, pad_))
#
#     mask = Image.new("L", (width, oh), 0)
#     mask = pad_image(mask, (0, 0, pad_, pad_), 255)
#
#     result = img2img_outpainting(
#         image=encode_pil_bytes(img, False),
#         mask=encode_pil_bytes(mask, False),
#         max_height=height,
#         max_width=width,
#         custom_prompt=line['gen_bg_prompt']
#     )
#     out_img = decode_pil_bytes(result, False)
#
# # xlsx_save(row_dict, "results_all_done_4567_v1.xlsx")
# print('done')



# # 主体分割
# from diffusers.data.clients.creative_ai_capability import image_subject_seg
# token = os.popen("doas -p ad.creative.image_core_solution printenv SEC_TOKEN_STRING|tail -1 ").read()
# os.environ["SEC_TOKEN_STRING"] = token  # os.environ['SEC_TOKEN_STRING'] = "toutiao.growth.xenon"
# if len(os.getenv("SEC_TOKEN_STRING", "")) < 1:
#     token = os.popen("cat /tmp/identity.token").read()
#     os.environ["SEC_TOKEN_STRING"] = token
# os.environ["SEC_KV_AUTH"] = "1"
#
# row_dict = load_csv_or_xlsx_to_dict("/mnt/bn/ttcc-algo-bytenas/zjn/data/poster_train/results_all_done_4567.csv")
#
# for line in tqdm(row_dict):
#     url = line['gen_url']
#     result = image_subject_seg(image_urls=[url], only_mask=1, refine_mask=0)
#     line['gen_mask_url'] = result.success_image_urls[0]
#
# xlsx_save(row_dict, "results_all_done_4567_v1.xlsx")
# print('done')





# 上传图像到tos
row_dict = load_csv_or_xlsx_to_dict("/mnt/bn/creative-algo/xsl/data/gpt_dataset/test_src.xlsx")

test_dir = "qwen_lora_test"
for line in tqdm(row_dict):
    name = basename(line['gen_url'])
    img_url = save_file_to_tos(join(test_dir, name + "_lora.jpg"), _gen_name('xsl'))
    line['infer_url'] = img_url

xlsx_save(row_dict, "test_src_v1.xlsx")
print('done')




# # html渲染
# from diffusers.data.render_html.render_html import renderUrl2CarouselSingleHtml
# row_dict = load_csv_or_xlsx_to_dict("/mnt/bn/creative-algo/xsl/data/gpt_dataset/test_src.xlsx")
# test_dir = "/mlx_devbox/users/xushangliang/playground/gpt_lora_test"
#
# ind = 0
# html_result = []
# for item in tqdm(row_dict):
#     html_item = {
#         "index": ind,
#         "product_name": item["text_product_names"],
#         "primary_selling_points": item["text_primary_selling_points"],
#         "secondary_selling_points": item["text_secondary_selling_points"],
#         "prompt": item["gpt_correction_prompt"],
#     }
#
#     upload_name = _gen_name("xsl")
#     for i in range(5):
#         img_url = save_file_to_tos(join(test_dir, basename(item['gen_url']) + f'_lora.jpg'), upload_name)
#         if img_url is not None:
#             break
#         else:
#             time.sleep(1)
#
#     html_item["ImageDiff"] = [
#         {
#             "title": "product",
#             "imageUrl": item['product_url'],
#         },
#         {
#             "title": "GT",
#             "imageUrl": item['gen_url'],
#         },
#         {
#             "title": "qwen_lora",
#             "imageUrl": img_url,
#         },
#     ]
#     html_result.append(html_item)
#     ind += 1
# for i in range(3):
#     code, msg, url = renderUrl2CarouselSingleHtml("/t2d_demo", f"xsl_exp_{time.time()}.html", json.dumps(html_result))
#     if url is not None:
#         break
#     else:
#         time.sleep(1)
# print(url)
# print('done')





# # OCR
# from diffusers.data.clients.creative_ai_capability import image_subject_seg
# from diffusers.data.clients.creative_ai_capability import image_ocr_to_bbox
# from overpass_ad_creative_ai_capabilities.euler_gen.idl.i18n_ad.creative.creative_factory.strategy.capability_thrift import ImageInfo
# from diffusers.metrics.ocr_word_metrics import calculate_word_accuracy, calculate_wer
#
# token = os.popen("doas -p ad.creative.image_core_solution printenv SEC_TOKEN_STRING|tail -1 ").read()
# os.environ["SEC_TOKEN_STRING"] = token  # os.environ['SEC_TOKEN_STRING'] = "toutiao.growth.xenon"
# if len(os.getenv("SEC_TOKEN_STRING", "")) < 1:
#     token = os.popen("cat /tmp/identity.token").read()
#     os.environ["SEC_TOKEN_STRING"] = token
# os.environ["SEC_KV_AUTH"] = "1"
#
#
# def get_image_text(url):
#     result = image_subject_seg(image_urls=[url], only_mask=1, refine_mask=0)
#     mask = load_or_download_image(result.success_image_urls[0])
#     img = load_or_download_image(url)
#     mask = np.array(mask.convert('L'))
#
#     mask = np.float32(mask[..., None]) / 255
#     bg_image = (1 - mask) * np.array(img, dtype=np.float32) + mask * 255
#     bg_image = Image.fromarray(np.uint8(np.clip(bg_image, 0, 255)))
#     image_info = ImageInfo(Binary=encode_pil_bytes(bg_image, False))
#
#     temp = []
#     for i in range(5):
#         if len(temp) > 0:
#             break
#         temp = image_ocr_to_bbox(image_infos=[image_info])[0]
#
#     pred_text = []
#     for item in temp:
#         pred_text.append(item['text'])
#     pred_text = " ".join(pred_text)
#     pred_text = pred_text.lower().replace(',', '').replace('，', '').replace('.', '')
#     return pred_text
#
# # 解析html中的data
# with open("xsl_exp_1755657489.3030906.html", "r", encoding="utf-8") as file:
#     html_content = file.read()
# html_content = html_content.split('var data = ')[1]
# html_content = html_content.split(';\n\n    /* ---------------- 兼容')[0]
# data = json.loads(html_content)
#
# avg_acc1 = []
# avg_wer1 = []
# avg_acc2 = []
# avg_wer2 = []
# for line in tqdm(data):
#     gt_text = get_image_text(line['ImageDiff'][1]['imageUrl'])
#     line['gt_text'] = gt_text
#     pred_text = get_image_text(line['ImageDiff'][2]['imageUrl'])
#     line['pred_text'] = pred_text
#     open_text = get_image_text(line['ImageDiff'][3]['imageUrl'])
#     line['open_text'] = open_text
#
#     if len(gt_text) > 0 and len(pred_text) > 0 and len(open_text) > 0:
#         avg_acc1.append(calculate_word_accuracy(pred_text, gt_text))
#         avg_wer1.append(calculate_wer(pred_text, gt_text))
#
#         avg_acc2.append(calculate_word_accuracy(open_text, gt_text))
#         avg_wer2.append(calculate_wer(open_text, gt_text))
#     else:
#         print(gt_text, pred_text, open_text)
#
# print(np.mean(avg_acc1), np.mean(avg_wer1))
# print(np.mean(avg_acc2), np.mean(avg_wer2))
# json_save(data, "ocr_results.json")
#
# print('done')
