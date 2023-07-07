import os
from os.path import join
import csv
import time
import json
from PIL import Image
from diffusers.data.render_html.render_html import renderUrl2CarouselSingleHtml
from diffusers.utils.vip_utils import load_image
from diffusers.data.utils import load_file, csv_save
from diffusers.data.outer_vos_tools import encode_pil_bytes
from diffusers.data.tos import save_tos, url_to_bytes, save_file_to_tos


# img_dir = "/mlx_devbox/users/xushangliang/playground/qwenimage"
# out_list = []
# for name in range(26):
#     name = f"{name}.jpg"
#     img_url = save_file_to_tos(join(img_dir, name), f"qwenimage_{name}")
#     out_list.append(img_url)
# csv_save(out_list, "qwenimage/urls.csv")

file_path = "/mlx_devbox/users/xushangliang/playground/creative_image_core_solution/whx_workspace/carousel/v7_folder/v7_results.csv"
html_result = []
with open(file_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for i, line in enumerate(reader):
        if line.get("result"):
            # item = json.loads(line.strip())
            item = line
            html_item = {}
            html_item["product_name"] = item.get("result_product_name")
            html_item["prompt"] = item.get("prompt")
            html_item["edit model"] = "https://amap-ml.github.io/FLUX-text/"
            html_item["ImageDiff"] = [
                {
                    "title": "source",
                    "imageUrl": item.get("target_url"),
                },
                {
                    "title": "flux_kontext",
                    "imageUrl": item.get("result"),
                },
                {
                    "title": "qwen_image",
                    "imageUrl": f"https://sf16-muse-va.ibytedtos.com/obj/ad-creative/qwenimage_{i}.jpg",
                },
            ]
            html_result.append(html_item)
code, msg, url = renderUrl2CarouselSingleHtml("/t2d_demo", f"qwen_{time.time()}.html", json.dumps(html_result))
print(url)
print('done')
