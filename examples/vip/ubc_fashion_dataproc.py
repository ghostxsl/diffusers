# Copyright (c) wilson.xu. All rights reserved.
import os
from os.path import join, splitext
from diffusers.data.videoreader import VideoReader
from PIL import Image
import csv

data_dir = "/xsl/wilson.xu/fashion_video"
split = ["train", "test"]
for split_dir in split:
    in_dir = join(data_dir, split_dir)
    out_dir = join(data_dir, f"{split_dir}_png")
    os.makedirs(out_dir, exist_ok=True)

    f = open(join(data_dir, f"{split_dir}_png.csv"), "a", encoding="utf-8", newline="")
    writer = csv.writer(f)
    writer.writerow(['file_name', 'text'])

    vid_list = sorted(os.listdir(in_dir))
    for i, name in enumerate(vid_list):
        video = VideoReader(join(in_dir, name))
        for j, img in enumerate(video.read()):
            Image.fromarray(img).save(join(out_dir, splitext(name)[0] + f'_{j}.png'))
            writer.writerow([splitext(name)[0] + f'_{j}.png', " "])
            f.flush()
        print(i, name)
    f.close()

print('Done!')
