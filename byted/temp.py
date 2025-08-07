import os
from os.path import join, splitext, basename
from tqdm import tqdm
import csv
import time
import json
import pandas as pd
from PIL import Image
from diffusers.data.render_html.render_html import renderUrl2CarouselSingleHtml
from diffusers.utils.vip_utils import load_image
from diffusers.data.utils import load_file, csv_save
from diffusers.data.outer_vos_tools import encode_pil_bytes
from diffusers.data.tos import url_to_bytes, save_file_to_tos
from diffusers.data.clients.creative_ai_capability import image_subject_seg


# 主体分割
token = os.popen("doas -p ad.creative.image_core_solution printenv SEC_TOKEN_STRING|tail -1 ").read()
os.environ["SEC_TOKEN_STRING"] = token  # os.environ['SEC_TOKEN_STRING'] = "toutiao.growth.xenon"
if len(os.getenv("SEC_TOKEN_STRING", "")) < 1:
    token = os.popen("cat /tmp/identity.token").read()
    os.environ["SEC_TOKEN_STRING"] = token
os.environ["SEC_KV_AUTH"] = "1"

df = pd.read_excel("/mnt/bn/ttcc-algo-bytenas/xsl/out1.xlsx")
row_dict = df.to_dict('records')

for line in tqdm(row_dict):
    url = line['bg_res_url']
    result = image_subject_seg(image_urls=[url], only_mask=1, refine_mask=0)
    line['bg_res_mask_url'] = result.success_image_urls[0]

df = pd.DataFrame(row_dict)
df.to_excel("out1.xlsx", index=False)
print('done')




# # 上传图像到tos
# file_path = "/mnt/bn/ttcc-algo-bytenas/xsl/out1.xlsx"
# df = pd.read_excel(file_path)
# row_dict = df.to_dict('records')

# img1_dir = "/mnt/bn/ttcc-algo-bytenas/xsl/output"
# img2_dir = "/mnt/bn/ttcc-algo-bytenas/xsl/output_flux_kontext_t2i"
# img3_dir = "/mnt/bn/ttcc-algo-bytenas/xsl/output_qwen"

# for line in tqdm(row_dict):
#     img_with_bg_url = line["bg_res_url"]
#     name = splitext(basename(img_with_bg_url))[0]

#     img_url = save_file_to_tos(join(img1_dir, name + "_kontext.jpg"), f"xsl/{name}_kontext.jpg")
#     kontext_t2i_url = save_file_to_tos(join(img2_dir, name + "_kontext_t2i.jpg"), f"xsl/{name}_kontext_t2i.jpg")
#     qwen_t2i_url = save_file_to_tos(join(img3_dir, name + "_qwen_t2i.jpg"), f"xsl/{name}_qwen_t2i.jpg")
#     line['kontext_url'] = img_url
#     line['kontext_t2i_url'] = kontext_t2i_url
#     line['qwen_t2i_url'] = qwen_t2i_url

# df = pd.DataFrame(row_dict)
# df.to_excel("out2.xlsx", index=False)
# print('done')



# # html渲染
# file_path = "out2.xlsx"
# df = pd.read_excel(file_path)
# row_dict = df.to_dict('records')

# html_result = []
# for item in tqdm(row_dict):
#     html_item = {
#         "product_name": item["result_product_name"],
#         "primary_selling_points": item["result_primary_selling_points"],
#         "secondary_selling_points": item["result_secondary_selling_points"],
#         "flux_kontext_prompt": item["flux_kontext_prompt"],
#     }

#     html_item["ImageDiff"] = [
#         {
#             "title": "source",
#             "imageUrl": item.get("source_url"),
#         },
#         {
#             "title": "change_bg",
#             "imageUrl": item.get("bg_res_url"),
#         },
#         {
#             "title": "flux_kontext",
#             "imageUrl": item.get("kontext_url"),
#         },
#         {
#             "title": "flux_kontext_t2i",
#             "imageUrl": item.get("kontext_t2i_url"),
#         },
#         {
#             "title": "qwen_image_t2i",
#             "imageUrl": item.get("qwen_t2i_url"),
#         },
#     ]
#     html_result.append(html_item)
# code, msg, url = renderUrl2CarouselSingleHtml("/t2d_demo", f"xsl_exp_{time.time()}.html", json.dumps(html_result))
# print(url)
# print('done')
