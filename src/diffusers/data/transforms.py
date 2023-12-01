# Copyright (c) wilson.xu. All rights reserved.
import numbers
from collections.abc import Sequence
import numpy as np
import cv2
from PIL import Image

import torch
import torchvision.transforms.functional as F
from torchvision.utils import _log_api_usage_once
from .utils import draw_bodypose, draw_handpose


__all__ = [
    'Resize', 'CenterCrop', 'RandomCrop',
    'RandomHorizontalFlip', 'RandomVerticalFlip',
    'ToTensor', 'Normalize', 'DrawPose',
    'ColorJitter', 'DrawCanny',
]


def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size


class Resize(torch.nn.Module):
    """Resize the input image to the given size.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

    .. warning::
        The output image might be different depending on its type: when downsampling, the interpolation of PIL images
        and tensors is slightly different, because PIL applies antialiasing. This may lead to significant differences
        in the performance of a network. Therefore, it is preferable to train and serve a model with the same input
        types. See also below the ``antialias`` parameter, which can help making the output of PIL images and tensors
        closer.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size).

            .. note::
                In torchscript mode size as single int is not supported, use a sequence of length 1: ``[size, ]``.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.BILINEAR``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.NEAREST_EXACT``,
            ``InterpolationMode.BILINEAR`` and ``InterpolationMode.BICUBIC`` are supported.
            The corresponding Pillow integer constants, e.g. ``PIL.Image.BILINEAR`` are accepted as well.
        max_size (int, optional): The maximum allowed for the longer edge of
            the resized image: if the longer edge of the image is greater
            than ``max_size`` after being resized according to ``size``, then
            the image is resized again so that the longer edge is equal to
            ``max_size``. As a result, ``size`` might be overruled, i.e. the
            smaller edge may be shorter than ``size``. This is only supported
            if ``size`` is an int (or a sequence of length 1 in torchscript
            mode).
        antialias (bool, optional): Whether to apply antialiasing.
            It only affects **tensors** with bilinear or bicubic modes and it is
            ignored otherwise: on PIL images, antialiasing is always applied on
            bilinear or bicubic modes; on other modes (for PIL images and
            tensors), antialiasing makes no sense and this parameter is ignored.
            Possible values are:

            - ``True``: will apply antialiasing for bilinear or bicubic modes.
              Other mode aren't affected. This is probably what you want to use.
            - ``False``: will not apply antialiasing for tensors on any mode. PIL
              images are still antialiased on bilinear or bicubic modes, because
              PIL doesn't support no antialias.
            - ``None``: equivalent to ``False`` for tensors and ``True`` for
              PIL images. This value exists for legacy reasons and you probably
              don't want to use it unless you really know what you are doing.

            The current default is ``None`` **but will change to** ``True`` **in
            v0.17** for the PIL and Tensor backends to be consistent.
    """

    def __init__(self,
                 size,
                 interpolation=F.InterpolationMode.BILINEAR,
                 max_size=None,
                 antialias="warn"):
        super().__init__()
        _log_api_usage_once(self)
        if not isinstance(size, (int, Sequence)):
            raise TypeError(f"Size should be int or sequence. Got {type(size)}")
        if isinstance(size, Sequence) and len(size) not in (1, 2):
            raise ValueError("If size is a sequence, it should have 1 or 2 values")
        self.size = size
        self.max_size = max_size
        if isinstance(interpolation, int):
            interpolation = F._interpolation_modes_from_int(interpolation)
        self.interpolation = interpolation
        self.antialias = antialias

    def forward(self, img):
        if isinstance(img, dict):
            for k, v in img.items():
                if 'image' in k:
                    img[k] = F.resize(v, self.size, self.interpolation, self.max_size, self.antialias)
        else:
            img = F.resize(img, self.size, self.interpolation, self.max_size, self.antialias)
        return img

    def __repr__(self) -> str:
        detail = f"(size={self.size}, interpolation={self.interpolation.value}, max_size={self.max_size}, antialias={self.antialias})"
        return f"{self.__class__.__name__}{detail}"


class CenterCrop(torch.nn.Module):
    """Crops the given image at the center.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
    If image size is smaller than output size along any edge, image is padded with 0 and then center cropped.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).
    """

    def __init__(self, size):
        super().__init__()
        _log_api_usage_once(self)
        self.size = tuple(_setup_size(size, error_msg="Please provide only two dimensions (h, w) for size."))

    def get_center_crop_params(self, img):
        """Crops the given image at the center.
        If the image is torch Tensor, it is expected
        to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
        If image size is smaller than output size along any edge, image is padded with 0 and then center cropped.

        Args:
            img (PIL Image or Tensor): Image to be cropped.
            output_size (sequence or int): (height, width) of the crop box. If int or sequence with single int,
                it is used for both directions.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        _, image_height, image_width = F.get_dimensions(img)
        crop_height, crop_width = self.size

        if crop_width > image_width or crop_height > image_height:
            padding_ltrb = [
                (crop_width - image_width) // 2 if crop_width > image_width else 0,
                (crop_height - image_height) // 2 if crop_height > image_height else 0,
                (crop_width - image_width + 1) // 2 if crop_width > image_width else 0,
                (crop_height - image_height + 1) // 2 if crop_height > image_height else 0,
            ]
            img = F.pad(img, padding_ltrb, fill=0)  # PIL uses fill value 0
            _, image_height, image_width = F.get_dimensions(img)
            if crop_width == image_width and crop_height == image_height:
                return img

        crop_top = int(round((image_height - crop_height) / 2.0))
        crop_left = int(round((image_width - crop_width) / 2.0))
        return [crop_top, crop_left, crop_height, crop_width]

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        if isinstance(img, dict):
            crop_params = self.get_center_crop_params(img['image'])
            for k, v in img.items():
                if 'image' in k:
                    img[k] = F.crop(v, *crop_params)
        else:
            crop_params = self.get_center_crop_params(img)
            img = F.crop(img, *crop_params)
        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"


class RandomCrop(torch.nn.Module):
    """Crop the given image at a random location.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions,
    but if non-constant padding is used, the input is expected to have at most 2 leading dimensions

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is None. If a single int is provided this
            is used to pad all borders. If sequence of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a sequence of length 4 is provided
            this is the padding for the left, top, right and bottom borders respectively.

            .. note::
                In torchscript mode padding as single int is not supported, use a sequence of
                length 1: ``[padding, ]``.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception. Since cropping is done
            after padding, the padding seems to be done at a random offset.
        fill (number or tuple): Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant.
            Only number is supported for torch Tensor.
            Only int or tuple value is supported for PIL Image.
        padding_mode (str): Type of padding. Should be: constant, edge, reflect or symmetric.
            Default is constant.

            - constant: pads with a constant value, this value is specified with fill

            - edge: pads with the last value at the edge of the image.
              If input a 5D torch Tensor, the last 3 dimensions will be padded instead of the last 2

            - reflect: pads with reflection of image without repeating the last value on the edge.
              For example, padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
              will result in [3, 2, 1, 2, 3, 4, 3, 2]

            - symmetric: pads with reflection of image repeating the last value on the edge.
              For example, padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
              will result in [2, 1, 1, 2, 3, 4, 4, 3]
    """
    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode="constant"):
        super().__init__()
        _log_api_usage_once(self)
        self.size = tuple(_setup_size(size, error_msg="Please provide only two dimensions (h, w) for size."))

        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    def _pad(self, img):
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

        _, height, width = F.get_dimensions(img)
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = F.pad(img, padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = F.pad(img, padding, self.fill, self.padding_mode)
        return img

    def get_params(self, img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image or Tensor): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        _, h, w = F.get_dimensions(img)
        th, tw = output_size

        if h < th or w < tw:
            raise ValueError(f"Required crop size {(th, tw)} is larger than input image size {(h, w)}")

        if w == tw and h == th:
            return 0, 0, h, w

        i = torch.randint(0, h - th + 1, size=(1,)).item()
        j = torch.randint(0, w - tw + 1, size=(1,)).item()
        return [i, j, th, tw]

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        if isinstance(img, dict):
            # padding image
            for k, v in img.items():
                if 'image' in k:
                    img[k] = self._pad(v)
            # random crop
            crop_params = self.get_params(img['image'], self.size)
            for k, v in img.items():
                if 'image' in k:
                    img[k] = F.crop(v, *crop_params)
        else:
            img = self._pad(img)
            crop_params = self.get_params(img, self.size)
            img = F.crop(img, *crop_params)
        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, padding={self.padding})"


class RandomHorizontalFlip(torch.nn.Module):
    """Horizontally flip the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        _log_api_usage_once(self)
        self.p = p

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        if torch.rand(1) < self.p:
            if isinstance(img, dict):
                for k, v in img.items():
                    if 'image' in k:
                        img[k] = F.hflip(v)
            else:
                img = F.hflip(img)
        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


class RandomVerticalFlip(torch.nn.Module):
    """Vertically flip the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        _log_api_usage_once(self)
        self.p = p

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        if torch.rand(1) < self.p:
            if isinstance(img, dict):
                for k, v in img.items():
                    if 'image' in k:
                        img[k] = F.vflip(v)
            else:
                img = F.vflip(img)
        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


class ToTensor(object):
    """Convert a PIL Image or ndarray to tensor and scale the values accordingly.

    This transform does not support torchscript.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8

    In the other cases, tensors are returned without scaling.

    .. note::
        Because the input image is scaled to [0.0, 1.0], this transformation should not be used when
        transforming target image masks. See the `references`_ for implementing the transforms for image masks.

    .. _references: https://github.com/pytorch/vision/tree/main/references/segmentation
    """

    def __init__(self) -> None:
        _log_api_usage_once(self)

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, dict):
            for k, v in pic.items():
                if 'image' in k:
                    pic[k] = F.to_tensor(v)
        else:
            pic = F.to_tensor(pic)
        return pic

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Normalize(torch.nn.Module):
    """Normalize a tensor image with mean and standard deviation.
    This transform does not support PIL Image.
    Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``
    channels, this transform will normalize each channel of the input
    ``torch.*Tensor`` i.e.,
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutate the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.

    """

    def __init__(self, mean, std, inplace=False):
        super().__init__()
        _log_api_usage_once(self)
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def forward(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        if isinstance(tensor, dict):
            for k, v in tensor.items():
                if 'image' in k:
                    tensor[k] = F.normalize(v, self.mean, self.std, self.inplace)
        else:
            tensor = F.normalize(tensor, self.mean, self.std, self.inplace)
        return tensor

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


class DrawPose(object):
    def __init__(self,
                 hand=True,
                 face=True,
                 body_kpt_thr=0.3,
                 hand_kpt_thr=0.3,
                 face_kpt_thr=0.3,):
        _log_api_usage_once(self)
        self.hand = hand
        self.face = face
        self.body_kpt_thr = body_kpt_thr
        self.hand_kpt_thr = hand_kpt_thr
        self.face_kpt_thr = face_kpt_thr

    def __call__(self, data):
        assert isinstance(data, dict)

        if 'condition' in data and 'image' in data:
            _, height, width = F.get_dimensions(data['image'])
            canvas = np.zeros(shape=(height, width, 3), dtype=np.uint8)
            kpts = data['condition']['body']['keypoints'][..., :2] * np.array([width, height])
            kpt_valid = data['condition']['body']['keypoints'][..., 2] > self.body_kpt_thr
            canvas = draw_bodypose(canvas, kpts, kpt_valid)
            if self.hand and data['condition']['hand'] is not None:
                kpts = data['condition']['hand']['keypoints'][..., :2] * np.array([width, height])
                kpt_valid = data['condition']['hand']['keypoints'][..., 2] > self.hand_kpt_thr
                canvas = draw_handpose(canvas, kpts, kpt_valid)
            data['condition_image'] = Image.fromarray(canvas)
        else:
            raise Exception(f"Not found keys: [`image`, `condition`]")

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class ColorJitter(torch.nn.Module):
    """Randomly change the brightness, contrast, saturation and hue of an image.
    If the image is torch Tensor, it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, mode "1", "I", "F" and modes with transparency (alpha channel) are not supported.

    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non-negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
            To jitter hue, the pixel values of the input image has to be non-negative for conversion to HSV space;
            thus it does not work if you normalize your image to an interval with negative values,
            or use an interpolation that generates negative values before using this function.
    """

    def __init__(
        self,
        brightness=0,
        contrast=0,
        saturation=0,
        hue=0,
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        self.brightness = self._check_input(brightness, "brightness")
        self.contrast = self._check_input(contrast, "contrast")
        self.saturation = self._check_input(saturation, "saturation")
        self.hue = self._check_input(hue, "hue", center=0, bound=(-0.5, 0.5), clip_first_on_zero=False)

    @torch.jit.unused
    def _check_input(self, value, name, center=1, bound=(0, float("inf")), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(f"If {name} is a single number, it must be non negative.")
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            value = [float(value[0]), float(value[1])]
        else:
            raise TypeError(f"{name} should be a single number or a list/tuple with length 2.")

        if not bound[0] <= value[0] <= value[1] <= bound[1]:
            raise ValueError(f"{name} values should be between {bound}, but got {value}.")

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            return None
        else:
            return tuple(value)

    @staticmethod
    def get_params(
        brightness,
        contrast,
        saturation,
        hue,
    ):
        """Get the parameters for the randomized transform to be applied on image.

        Args:
            brightness (tuple of float (min, max), optional): The range from which the brightness_factor is chosen
                uniformly. Pass None to turn off the transformation.
            contrast (tuple of float (min, max), optional): The range from which the contrast_factor is chosen
                uniformly. Pass None to turn off the transformation.
            saturation (tuple of float (min, max), optional): The range from which the saturation_factor is chosen
                uniformly. Pass None to turn off the transformation.
            hue (tuple of float (min, max), optional): The range from which the hue_factor is chosen uniformly.
                Pass None to turn off the transformation.

        Returns:
            tuple: The parameters used to apply the randomized transform
            along with their random order.
        """
        fn_idx = torch.randperm(4)

        b = None if brightness is None else float(torch.empty(1).uniform_(brightness[0], brightness[1]))
        c = None if contrast is None else float(torch.empty(1).uniform_(contrast[0], contrast[1]))
        s = None if saturation is None else float(torch.empty(1).uniform_(saturation[0], saturation[1]))
        h = None if hue is None else float(torch.empty(1).uniform_(hue[0], hue[1]))

        return fn_idx, b, c, s, h

    def forward(self, imgs):
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = self.get_params(
            self.brightness, self.contrast, self.saturation, self.hue
        )

        if isinstance(imgs, dict):
            img = imgs['image']
        else:
            img = imgs

        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                img = F.adjust_brightness(img, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                img = F.adjust_contrast(img, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                img = F.adjust_saturation(img, saturation_factor)
            elif fn_id == 3 and hue_factor is not None:
                img = F.adjust_hue(img, hue_factor)

        if isinstance(imgs, dict):
            imgs['image'] = img
            imgs['condition_image'] = self.get_color_palette(img, imgs['condition_image'])
        else:
            imgs = img

        return imgs

    def get_color_palette(self,
                          image,
                          mask,
                          color_code={
                              'hair': 2,
                              'neck': 10,
                              'face': 13,
                              'left_arm': 14,
                              'right_arm': 15,
                              'left_leg': 16,
                              'right_leg': 17,
                          }):
        image = np.array(image, dtype=np.float32)
        mask = np.array(mask, dtype=np.float32)
        out = np.full_like(image, 255.)
        for k, v in color_code.items():
            temp_mask = np.zeros_like(mask)
            temp_mask[mask == v] = 1
            mean_val = np.sum(image * temp_mask) / np.sum(temp_mask)
            out[temp_mask[..., None] == 1] = mean_val
        return Image.fromarray(out).convert("F")

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"brightness={self.brightness}"
            f", contrast={self.contrast}"
            f", saturation={self.saturation}"
            f", hue={self.hue})"
        )
        return s


class DrawCanny(object):
    def __init__(self, thr_a=[100, 50], thr_b=[200, 150]):
        _log_api_usage_once(self)
        self.thr_a = thr_a
        self.thr_b = thr_b

    def __call__(self, data):
        assert isinstance(data, dict)

        if 'image' in data:
            img = np.array(data['image'])
            canny = None
            for a, b in zip(self.thr_a, self.thr_b):
                if canny is None:
                    canny = cv2.Canny(img, a, b).astype('bool')
                else:
                    canny = np.logical_or(canny, cv2.Canny(img, a, b).astype('bool'))

            canny = canny.astype('uint8') * 255
            canny = np.tile(canny[..., None], [1, 1, 3])
            data['condition_image'] = Image.fromarray(canny)
        else:
            raise Exception(f"Not found keys: [`image`]")

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
