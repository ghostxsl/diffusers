import os
from os.path import join, splitext
import numpy as np
import cv2
from PIL import Image
from collections import OrderedDict
import pickle
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from mmpose.blendpose.utils import pkl_save, pkl_load



bg_dict = pkl_load("bg_hist.pkl")
bg_name = list(bg_dict['hist'].keys())
bg_val = np.stack(list(bg_dict['hist_fit'].values()))
# high = np.percentile(bg_val, 99.5, axis=1)
# low = np.percentile(bg_val, 1, axis=1)

fg_dir = "/xsl/wilson.xu/dataset/tmp_imgs/fg_img"
bg_dir = "/xsl/wilson.xu/dataset/tmp_imgs/bg_img"
out_dir = "/xsl/wilson.xu/dataset/tmp_imgs/match_img"
fg_list = os.listdir(fg_dir)
fg_dict = OrderedDict()
bins = list(range(257))
for i, name in enumerate(fg_list):
    img = Image.open(join(fg_dir, name)).convert("RGB")
    img_size = img.size
    img = np.array(img)
    L_img = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)[..., 0]
    hist, bin_edges = np.histogram(L_img, bins=bins, normed=True)
    func = interp1d(bins[:-1], hist, kind='cubic')
    x = np.linspace(5, 250, num=100)
    y = func(x)
    # plt.figure()
    # plt.plot(bins[:-1], hist, 'o', x, y, '-')
    # plt.savefig("hist_fg.png")
    # plt.close()

    dis = np.linalg.norm(bg_val - y, axis=1)
    min_name = bg_name[dis.argmin()]
    bg_hist = bg_dict['hist'][min_name]
    yy = bg_dict['hist_fit'][min_name]
    # plt.figure()
    # plt.plot(bins[:-1], bg_hist, 'o', x, yy, '-')
    # plt.savefig("hist_bg.png")
    # plt.close()

    fg_dict[name] = {"hist": hist, "hist_fit": y, "bg_name": min_name}
    print(i, name)
    bg_img = np.array(Image.open(join(bg_dir, min_name)).convert("RGB").resize(img_size, 1))
    Image.fromarray(np.concatenate([img, bg_img], axis=1)).save(join(out_dir, f"{splitext(name)[0]}_{splitext(min_name)[0]}.jpg"))

pkl_save(fg_dict, "fg_match.pkl")
print("Done")




# fg_dir = "/xsl/wilson.xu/dataset/tmp_imgs/fg_img"
# bg_dir = "/xsl/wilson.xu/dataset/tmp_imgs/bg_img"

# # fg_list = os.listdir(fg_dir)
# bg_list = os.listdir(bg_dir)
# bg_dict = OrderedDict()
# bg_dict["hist"] = {}
# bg_dict["hist_fit"] = {}
# bins = list(range(257))
# for i, name in enumerate(bg_list):
#     img = np.array(Image.open(join(bg_dir, name)).convert("RGB"))
#     L_img = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)[..., 0]
#     hist, bin_edges = np.histogram(L_img, bins=bins, normed=True)
#     func = interp1d(bins[:-1], hist, kind='cubic')
#     x = np.linspace(5, 250, num=100)
#     y = func(x)
#     # plt.figure()
#     # plt.plot(bins[:-1], hist, 'o', x, y, '-')
#     # plt.savefig("hist_fg1.png")
#     # plt.close()
#     bg_dict["hist"][name] = hist
#     bg_dict["hist_fit"][name] = y
#     print(i, name)

# pkl_save(bg_dict, "bg_hist.pkl")
# print("Done")
