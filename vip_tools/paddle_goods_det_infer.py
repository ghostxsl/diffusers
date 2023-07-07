# coding: utf-8
# author: zuchen.wang@vipshop.com
# code refactored from http://gitlab.tools.vipshop.com/guang01.yao/detect_image_infer/master
# @author: wilson.xu.

import os
from os.path import join, splitext, split
import yaml
import argparse

from tqdm import tqdm
import paddle
import paddle.inference as paddle_infer
import numpy as np
import cv2
from PIL import Image

from diffusers.data.utils import *
from diffusers.utils.vip_utils import load_image
from diffusers.data.vos_client import VOSClient


def parse_args():
    parser = argparse.ArgumentParser(description="infer script.")
    parser.add_argument(
        "--img_dir",
        default=None,
        type=str,
        help="Directory to image.")
    parser.add_argument(
        "--vos_pkl",
        default=None,
        type=str,
        help="Path to image list on vos.")
    parser.add_argument(
        "--out_file",
        default="output.pkl",
        type=str,
        help="File path to save.")
    parser.add_argument(
        "--model_path",
        default="/apps/dat/cv/xsl/aistudio_weights/extensions/paddle_goods_det",
        type=str,
        help="Directory to weights.")

    parser.add_argument(
        "--rank",
        default=None,
        type=int)
    parser.add_argument(
        "--num_ranks",
        default=None,
        type=int)

    args = parser.parse_args()

    return args


class Resize(object):
    """resize image by target_size and max_size
    Args:
        target_size (int): the target size of image
        keep_ratio (bool): whether keep_ratio or not, default true
        interp (int): method of resize
    """

    def __init__(self, target_size, keep_ratio=True, interp=cv2.INTER_LINEAR):
        if isinstance(target_size, int):
            target_size = [target_size, target_size]

        assert len(target_size) == 2
        assert target_size[0] > 0 and target_size[1] > 0

        self.target_size = target_size
        self.keep_ratio = keep_ratio
        self.interp = interp

    def __call__(self, im, im_info):
        """
        Args:
            im (np.ndarray): image (np.ndarray)
            im_info (dict): info of image
        Returns:
            im (np.ndarray):  processed image (np.ndarray)
            im_info (dict): info of processed image
        """
        im_channel = im.shape[2]
        im_scale_y, im_scale_x = self.generate_scale(im)
        im = cv2.resize(im, None, None, fx=im_scale_x, fy=im_scale_y, interpolation=self.interp)
        im_info["img_hw"] = np.array(im.shape[:2]).astype("float32")
        im_info["scale_factor"] = np.array([im_scale_y, im_scale_x]).astype("float32")
        return im, im_info

    def generate_scale(self, im):
        """
        Args:
            im (np.ndarray): image (np.ndarray)
        Returns:
            im_scale_x: the resize ratio of X
            im_scale_y: the resize ratio of Y
        """
        origin_shape = im.shape[:2]
        im_c = im.shape[2]
        if self.keep_ratio:
            im_size_min = np.min(origin_shape)
            im_size_max = np.max(origin_shape)
            target_size_min = np.min(self.target_size)
            target_size_max = np.max(self.target_size)
            im_scale = float(target_size_min) / float(im_size_min)
            if np.round(im_scale * im_size_max) > target_size_max:
                im_scale = float(target_size_max) / float(im_size_max)
            im_scale_x = im_scale
            im_scale_y = im_scale
        else:
            resize_h, resize_w = self.target_size
            im_scale_y = resize_h / float(origin_shape[0])
            im_scale_x = resize_w / float(origin_shape[1])
        return im_scale_y, im_scale_x


class NormalizeImage(object):
    """normalize image
    Args:
        mean (list): im - mean
        std (list): im / std
        is_scale (bool): whether need im / 255
        norm_type (str): type in ['mean_std', 'none']
    """

    def __init__(self, mean, std, is_scale=True, norm_type="mean_std"):
        self.mean = mean
        self.std = std
        self.is_scale = is_scale
        self.norm_type = norm_type

    def __call__(self, im, im_info):
        """
        Args:
            im (np.ndarray): image (np.ndarray)
            im_info (dict): info of image
        Returns:
            im (np.ndarray):  processed image (np.ndarray)
            im_info (dict): info of processed image
        """
        im = im.astype(np.float32, copy=False)
        if self.is_scale:
            scale = 1.0 / 255.0
            im *= scale

        if self.norm_type == "mean_std":
            mean = np.array(self.mean)[np.newaxis, np.newaxis, :]
            std = np.array(self.std)[np.newaxis, np.newaxis, :]
            im -= mean
            im /= std
        return im, im_info


class Permute(object):
    """permute image
    Args:
        to_bgr (bool): whether convert RGB to BGR
        channel_first (bool): whether convert HWC to CHW
    """

    def __init__(self, ):
        super(Permute, self).__init__()

    def __call__(self, im, im_info):
        """
        Args:
            im (np.ndarray): image (np.ndarray)
            im_info (dict): info of image
        Returns:
            im (np.ndarray):  processed image (np.ndarray)
            im_info (dict): info of processed image
        """
        im = im.transpose((2, 0, 1)).copy()
        return im, im_info


class Detector(object):
    def __init__(self,
                 model_path="paddle_goods_det",
                 box_thresh=0.45,
                 keep_cates=('bag', 'electronics')):
        self.box_thresh = box_thresh
        self.keep_cates = set(keep_cates)

        yml_cfg_file = join(model_path, "infer_cfg_pzg.yml")
        print(f"Load cfg file: {yml_cfg_file}")
        self.yml_cfg = yaml.safe_load(open(yml_cfg_file))
        self.label_dict = {idx: cate for idx, cate in enumerate(self.yml_cfg["label_list"])}
        self.image_transforms = []
        for op_info in self.yml_cfg["Preprocess"]:
            _op_info = op_info.copy()
            _op_type = _op_info.pop("type")
            self.image_transforms.append(eval(_op_type)(**_op_info))
        print(f"image transforms: {self.image_transforms}")

        model_file = join(model_path, "model.pdmodel")
        params_file = join(model_path, "model.pdiparams")
        self.config = paddle_infer.Config(model_file, params_file)
        self.config.enable_use_gpu(100, 0)  # 100MB显存, gpu id: 0
        self.config.switch_ir_optim(False)
        print(f"Model Config:\n{self.config.summary()}")
        self.predictor = paddle.inference.create_predictor(self.config)
        self.predictor_input_names = self.predictor.get_input_names()
        self.predictor_output_names = self.predictor.get_output_names()

    def read_image(self, args, name):
        if args.vos_pkl is None:
            img = load_image(join(args.img_dir, name))
        else:
            img = args.vos_client.download_vos_pil(name)
            img = load_image(img)

        return img

    def __call__(self, img_paths, args):
        if isinstance(img_paths, str):
            img_paths = [img_paths]
        img_list = []
        for name in img_paths:
            img_list.append(np.array(self.read_image(args, name)))

        batch_inputs = self.pre_process(img_list)
        for name in self.predictor_input_names:
            input_handle = self.predictor.get_input_handle(name)
            input_handle.copy_from_cpu(batch_inputs[name])

        self.predictor.run()

        batch_box_info = self.predictor.get_output_handle(self.predictor_output_names[0]).copy_to_cpu()
        batch_box_num = self.predictor.get_output_handle(self.predictor_output_names[1]).copy_to_cpu()
        batch_img_shape = batch_inputs["img_hw"] / batch_inputs["scale_factor"]
        batch_img_shape = batch_img_shape.astype(np.int32)

        batch_cates, batch_confidences, batch_boxes = self.post_process(batch_box_info, batch_box_num, batch_img_shape)

        results = []
        for img_path, cates, confidences, boxes, img_hw in zip(img_paths, batch_cates, batch_confidences, batch_boxes,
                                                               batch_img_shape):
            result = {"img_path": img_path, "img_hw": img_hw.tolist(), "boxes": []}
            for cate, conf, box in zip(cates, confidences, boxes):
                result["boxes"].append({"category": cate, "confidence": conf.item(), "box": box.tolist()})
            results.append(result)

        return results

    def pre_process(cls, img_list):
        batch_img = []
        batch_pad_img = []
        batch_img_hw = []
        batch_scale_factor = []
        for img in img_list:
            img_height, img_width = img.shape[:2]
            img_info = {
                "img_hw": np.array((img_height, img_width), dtype=np.float32),
                "scale_factor": np.ones((2, ), dtype=np.float32),
            }
            for trans in cls.image_transforms:
                img, img_info = trans(img, img_info)
            batch_img.append(img)
            batch_img_hw.append(img_info["img_hw"])
            batch_scale_factor.append(img_info["scale_factor"])

        batch_img_hw = np.stack(batch_img_hw, axis=0)
        batch_scale_factor = np.stack(batch_scale_factor, axis=0)

        max_batch_height = np.max(batch_img_hw[:, 0]).astype(int)
        max_batch_width = np.max(batch_img_hw[:, 1]).astype(int)

        for img in batch_img:
            img_channel, img_height, img_width = img.shape
            if img_height == max_batch_height and img_width == max_batch_width:
                batch_pad_img.append(img)
            else:
                pad_img = np.zeros((img_channel, max_batch_height, max_batch_width), dtype=np.float32)
                pad_img[:, :img_height, :img_width] = img
                batch_pad_img.append(pad_img)
        batch_pad_img = np.stack(batch_pad_img, axis=0)
        return {
            "image": batch_pad_img,
            "img_hw": batch_img_hw,
            "scale_factor": batch_scale_factor,
        }

    def post_process(cls, batch_box_info, batch_box_num, batch_img_shape):
        batch_cates = []
        batch_confs = []
        batch_boxes = []

        if batch_box_num.sum() == 0:
            print("Warning: this batch was detected 0 box")
        else:
            prev_cum_num = 0
            for i, num in enumerate(batch_box_num):
                cur_cum_num = prev_cum_num + num
                boxes_info = batch_box_info[prev_cum_num:prev_cum_num + num, :]
                prev_cum_num = cur_cum_num
                # boxes: cate, confidence, x, y, x, y
                cates = boxes_info[:, 0]
                confs = boxes_info[:, 1]
                boxes = boxes_info[:, 2:]

                idxs = np.argwhere(confs > cls.box_thresh).flatten()
                # 再对指定类别做一次过滤
                idxs = [i for i in idxs if cls.label_dict[cates[i]] in cls.keep_cates]
                cates = cates[idxs].astype(int)
                cates = [cls.label_dict[x] for x in cates]
                confs = confs[idxs]
                boxes = boxes[idxs]

                # 再做一次nms
                if len(boxes) >= 2:
                    nms_idxs = paddle.vision.ops.nms(paddle.to_tensor(boxes, place='cpu'),
                                                     iou_threshold=0.3,
                                                     scores=paddle.to_tensor(confs, place='cpu')).numpy()
                    cates = [cates[i] for i in nms_idxs]
                    confs = confs[nms_idxs]
                    boxes = boxes[nms_idxs]

                boxes = np.maximum(boxes, 0)
                boxes[:, 0:3:2] = np.minimum(boxes[:, 0:3:2], batch_img_shape[i, 1])
                boxes[:, 1:4:2] = np.minimum(boxes[:, 1:4:2], batch_img_shape[i, 0])
                boxes = boxes + 0.5
                boxes = boxes.astype(int)

                batch_cates.append(cates)
                batch_confs.append(confs)
                batch_boxes.append(boxes)

        return batch_cates, batch_confs, batch_boxes


if __name__ == '__main__':
    args = parse_args()
    detector = Detector(model_path=args.model_path)

    if args.vos_pkl is None:
        img_list = sorted(os.listdir(args.img_dir))
    else:
        img_list = load_file(args.vos_pkl)
        args.vos_client = VOSClient()

    if args.rank is not None and args.num_ranks is not None:
        assert args.rank < args.num_ranks
        total_num = len(img_list)
        stride = int(total_num / args.num_ranks)

        start_idx = stride * args.rank
        end_idx = stride * (args.rank + 1) if args.rank + 1 < args.num_ranks else len(img_list)
        img_list = img_list[start_idx: end_idx]

    results = []
    for img_path in tqdm(img_list):
        res = detector(img_path, args)
        results.extend(res)

    if args.rank is not None:
        out_dir, out_file = split(args.out_file)
        out_file = splitext(out_file)[0] + f"_{args.rank}.pkl"
        pkl_save(results, join(out_dir, out_file))
    else:
        pkl_save(results, args.out_file)

    print('done')
