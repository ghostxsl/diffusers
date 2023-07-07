import os
from os.path import join, splitext
import csv

img_dir = "/xsl/wilson.xu/dataset/hair_man_train/5_short"
csv_file = "/xsl/wilson.xu/dataset/hair_man_train/5_short.csv"

f = open(csv_file, "a", encoding="utf-8", newline="")
writer = csv.writer(f)
writer.writerow(['file_name', 'text'])

img_list = os.listdir(img_dir)

for i, name in enumerate(img_list):
    print(i, name)
    writer.writerow([name, "short_hairstyle, man posing for a photo, simple background"])
    f.flush()

f.close()
print('Done!')
