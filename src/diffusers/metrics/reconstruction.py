
import os
from os.path import exists, splitext, join
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr


__all__ = ['ReconstructionMetric']


def compare_l1(img_true, img_test):
    img_true = img_true.astype(np.float32)
    img_test = img_test.astype(np.float32)
    return np.mean(np.abs(img_true - img_test))


def compare_mae(img_true, img_test):
    img_true = img_true.astype(np.float32)
    img_test = img_test.astype(np.float32)
    return np.sum(np.abs(img_true - img_test)) / np.sum(img_true + img_test)


class ReconstructionMetric(object):

    def __init__(
            self,
            metric_list=['ssim', 'psnr', 'l1', 'mae'],
            data_range=1,
            win_size=51):
        self.data_range = data_range
        self.win_size = win_size
        self.metric_list = metric_list

    def __call__(self, pred_img, gt_img):
        """
        input_img: the generated image, size (h,w,c), data range(0, data_range)
        gt_img:    the ground-truth image, size (h,w,c), data range(0, data_range)
        """
        pred_img = np.array(pred_img).astype(np.float32)
        gt_img = np.array(gt_img).astype(np.float32)

        result = {}
        if "ssim" in self.metric_list:
            result['ssim'] = compare_ssim(
                pred_img, gt_img,
                win_size=self.win_size,
                data_range=self.data_range,
                channel_axis=2,
            )

            result['ssim_gaussian'] = compare_ssim(
                pred_img, gt_img,
                data_range=self.data_range,
                channel_axis=2,
                gaussian_weights=True,
                sigma=1.2,
                use_sample_covariance=False,
            )

        if "psnr" in self.metric_list:
            result['psnr'] = compare_psnr(
                gt_img, pred_img, data_range=self.data_range)

        if "l1" in self.metric_list:
            result['l1'] = compare_l1(gt_img, pred_img)

        if "mae" in self.metric_list:
            result['mae'] = compare_mae(gt_img, pred_img)

        return result
