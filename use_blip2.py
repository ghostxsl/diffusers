from lavis.models import load_model_and_preprocess
from PIL import Image
import torch
import os
from os.path import join, splitext
import csv
import shutil


img_dir = "/xsl/wilson.xu/dataset/tmp_imgs/20230721_wilson.xu_50w.csv"
out_dir = "/xsl/wilson.xu/dataset/train_images"
out_file = "/xsl/wilson.xu/dataset/42w_caption.csv"
device = torch.device("cuda")

model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_opt", model_type="caption_coco_opt2.7b", is_eval=True, device=device
)

f = open(out_file, "a", encoding="utf-8", newline="")
writer = csv.writer(f)
# writer.writerow(['file_name', 'text'])

num_no_caption = 0
img_list = os.listdir(img_dir)
for i, name in enumerate(img_list):
    print(f"{i + 1}/{len(img_list)}: {name}")
    if splitext(name)[-1].lower() in ['.jpg', '.jpeg', '.png']:
        img = Image.open(join(img_dir, name)).convert("RGB")
        image = vis_processors["eval"](img).unsqueeze(0).to(device)
        caption = model.generate({"image": image})
        if caption[0].strip() == '':
            num_no_caption += 1
            print(f"No caption: {name}/{num_no_caption}")
            continue
        else:
            writer.writerow([name, caption[0]])
            f.flush()
            print(f"caption: {caption[0]}")
            src = join(img_dir, name)
            dst = join(out_dir, name)
            shutil.move(src, dst)
    else:
        print(f"Wrong type: {name}")

f.close()
print("Done")
