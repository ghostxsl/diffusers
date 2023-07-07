# Copyright (c) wilson.xu. All rights reserved.
import numbers
import random
from collections.abc import Sequence
import numpy as np
import cv2
from PIL import Image

import torch
import torchvision
import torchvision.transforms.functional as F
from torchvision.utils import _log_api_usage_once
from .utils import draw_bodypose, draw_handpose, draw_facepose, crop_human_bbox


__all__ = [
    'Resize', 'RandomResize', 'CenterCrop', 'RandomCrop',
    'RandomHorizontalFlip', 'RandomVerticalFlip',
    'ToTensor', 'Normalize', 'DrawPose',
    'ColorJitter', 'DrawCanny', 'ResizePad',
    'PasteMatting', 'BoxCrop', 'HumanCrop',
    'RandomSelects', 'RandomRectMask', 'RandomHumanMask',
    'PTRandomCrop', 'FLUXICFillProc', 'TextPTRandomCrop',
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


class RandomResize(torch.nn.Module):
    def __init__(self,
                 size,
                 random_size,
                 prob=0.5,
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

        assert isinstance(random_size, Sequence) and len(random_size) == 2
        self.random_size = random_size

        self.prob = prob
        self.max_size = max_size
        if isinstance(interpolation, int):
            interpolation = F._interpolation_modes_from_int(interpolation)
        self.interpolation = interpolation
        self.antialias = antialias

    def forward(self, img):
        size = random.randint(*self.random_size) if random.random() <= self.prob else self.size

        if isinstance(img, dict):
            for k, v in img.items():
                if 'image' in k:
                    img[k] = F.resize(v, size, self.interpolation, self.max_size, self.antialias)
        else:
            img = F.resize(img, size, self.interpolation, self.max_size, self.antialias)
        return img

    def __repr__(self) -> str:
        detail = f"(size={self.size}, random_size={self.random_size}, prob={self.prob})"
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

    def _pad(self, img, fill=0):
        if self.padding is not None:
            img = F.pad(img, self.padding, fill, self.padding_mode)

        _, height, width = F.get_dimensions(img)
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [0, 0, self.size[1] - width, 0]
            img = F.pad(img, padding, fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, 0, 0, self.size[0] - height]
            img = F.pad(img, padding, fill, self.padding_mode)
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
                    fill = 0 if k == 'condition_image' else self.fill
                    img[k] = self._pad(v, fill)
            # random crop
            crop_params = self.get_params(img['image'], self.size)
            for k, v in img.items():
                if 'image' in k:
                    img[k] = F.crop(v, *crop_params)
        else:
            img = self._pad(img, self.fill)
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
                        if isinstance(v, Sequence):
                            v = [F.hflip(p) for p in v]
                            img[k] = v
                        else:
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
                 size=1024,
                 prob_hand=0.5,
                 prob_face=0.5,
                 body_kpt_thr=0.3,
                 hand_kpt_thr=0.3,
                 face_kpt_thr=0.3,
                 random_ratio=None):
        _log_api_usage_once(self)
        self.size = size
        self.prob_hand = prob_hand
        self.prob_face = prob_face
        self.body_kpt_thr = body_kpt_thr
        self.hand_kpt_thr = hand_kpt_thr
        self.face_kpt_thr = face_kpt_thr
        self.random_ratio = random_ratio

    def draw_pose(self, img_size, points_dict):
        height, width = img_size
        size = height if self.size is None else self.size

        if self.random_ratio is not None and random.random() < 0.5:
            a, b = self.random_ratio
            size = random.randint(int(size * a), int(size * b))

        w = int(size / height * width) if size != height else width
        canvas = np.zeros(shape=(size, w, 3), dtype=np.uint8)
        size_ = np.array([w, size])

        kpts = points_dict['body']['keypoints'][..., :2] * size_
        kpt_valid = points_dict['body']['keypoints'][..., 2] > self.body_kpt_thr
        canvas = draw_bodypose(canvas, kpts, kpt_valid)

        if points_dict['hand'] is not None and random.random() <= self.prob_hand:
            kpts = points_dict['hand']['keypoints'][..., :2] * size_
            kpt_valid = points_dict['hand']['keypoints'][..., 2] > self.hand_kpt_thr
            canvas = draw_handpose(canvas, kpts, kpt_valid)

        if points_dict['face'] is not None and random.random() <= self.prob_face:
            kpts = points_dict['face']['keypoints'][..., :2] * size_
            kpt_valid = points_dict['face']['keypoints'][..., 2] > self.face_kpt_thr
            canvas = draw_facepose(canvas, kpts, kpt_valid)

        return Image.fromarray(canvas).resize((width, height), Image.Resampling.NEAREST)

    def __call__(self, data):
        assert isinstance(data, dict)

        if 'condition' in data and 'image' in data:
            if isinstance(data['image'], Sequence) and isinstance(data['condition'], Sequence):
                out_cond = []
                for img, cond in zip(data['image'], data['condition']):
                    _, height, width = F.get_dimensions(img)
                    out_cond.append(self.draw_pose((height, width), cond))
                data['condition_image'] = out_cond
            else:
                _, height, width = F.get_dimensions(data['image'])
                data['condition_image'] = self.draw_pose((height, width), data['condition'])
        else:
            raise Exception(f"Not found keys: [`image`, `condition`]")

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(draw_hand: {self.prob_hand}, draw_face: {self.prob_face})"


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
        elif isinstance(value, Sequence) and len(value) == 2:
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


class ResizePad(object):
    def __init__(self,
                 size,
                 interpolation='bilinear',
                 padding=True,
                 pad_values=255):
        super().__init__()
        _log_api_usage_once(self)
        if not isinstance(size, (int, Sequence)):
            raise TypeError(f"Size should be int or sequence. Got {type(size)}")
        if isinstance(size, Sequence) and len(size) != 2:
            raise ValueError("If size is a sequence, it should have 2 values")
        self.size = size
        self.interpolation = interpolation
        self.padding = padding
        self.pad_values = pad_values

    def resize_and_pad_pil(self, img, pad_values=255):
        w, h = img.size
        # resize
        if isinstance(self.size, int):
            if w > h:
                h = int(self.size / w * h)
                w = self.size
            elif h > w:
                w = int(self.size / h * w)
                h = self.size
            else:
                w = h = self.size
        elif isinstance(self.size, Sequence):
            h, w = self.size

        img = img.resize((w, h), resample=Image.Resampling.LANCZOS)

        if self.padding:
            # pad
            img = np.array(img)
            if w > h:
                pad_ = w - h
                img = np.pad(
                    img,
                    ((pad_ // 2, pad_ - pad_ // 2), (0, 0), (0, 0)),
                    constant_values=pad_values
                )
            elif h > w:
                pad_ = h - w
                img = np.pad(
                    img,
                    ((0, 0), (pad_ // 2, pad_ - pad_ // 2), (0, 0)),
                    constant_values=pad_values
                )
            img = Image.fromarray(img)

        return img

    def resize_and_pad_tensor(self, img, pad_values=255):
        h, w = img.shape[-2:]
        # resize
        if isinstance(self.size, int):
            if w > h:
                h = int(self.size / w * h)
                w = self.size
            elif h > w:
                w = int(self.size / h * w)
                h = self.size
            else:
                w = h = self.size
        elif isinstance(self.size, Sequence):
            h, w = self.size

        if img.ndim == 3:
            img = img.unsqueeze(0)
            img = torch.nn.functional.interpolate(
                img,
                size=[h, w],
                mode=self.interpolation,
                antialias=True
            )
            img = img.squeeze(0)
        elif img.ndim == 4:
            img = torch.nn.functional.interpolate(
                img,
                size=[h, w],
                mode=self.interpolation,
                antialias=True
            )

        if self.padding:
            # pad
            if w > h:
                pad_ = w - h
                img = torch.nn.functional.pad(
                    img,
                    (0, 0, pad_ // 2, pad_ - pad_ // 2),
                    value=pad_values / 255.
                )
            elif h > w:
                pad_ = h - w
                img = torch.nn.functional.pad(
                    img,
                    (pad_ // 2, pad_ - pad_ // 2, 0, 0),
                    value=pad_values / 255.
                )

        return img

    def __call__(self, img):
        if isinstance(img, dict):
            for k, v in img.items():
                if 'image' in k:
                    if isinstance(v, Image.Image):
                        img[k] = self.resize_and_pad_pil(
                            v, self.pad_values if k != 'condition_image' else 0)
                    elif isinstance(v, torch.Tensor):
                        img[k] = self.resize_and_pad_tensor(
                            v, self.pad_values if k != 'condition_image' else 0)
                    elif isinstance(v, Sequence) and isinstance(v[0], Image.Image):
                        v = [
                            self.resize_and_pad_pil(
                                p, self.pad_values if k != 'condition_image' else 0) for p in v
                        ]
                        img[k] = v
        else:
            if isinstance(img, Image.Image):
                img = self.resize_and_pad_pil(img, self.pad_values)
            elif isinstance(img, torch.Tensor):
                img = self.resize_and_pad_tensor(img, self.pad_values)
        return img

    def __repr__(self) -> str:
        detail = f"(size={self.size}, interpolation={self.interpolation}"
        return f"{self.__class__.__name__}{detail}"


class PasteMatting(object):
    def __init__(self, bg_value=255):
        _log_api_usage_once(self)
        if isinstance(bg_value, (int, float)):
            bg_value = int(bg_value)
            bg_value = [bg_value, ] * 3
        elif isinstance(bg_value, Sequence):
            assert len(bg_value) == 3
        else:
            raise Exception(f"Error bg_value type: {type(bg_value)}")

        self.bg_value = bg_value

    def paste_matting(self, src_img, matting):
        _, height, width = F.get_dimensions(src_img)
        bg = np.ones(shape=(height, width, 3), dtype=np.float32) * np.array(self.bg_value)

        src_img = np.array(src_img).astype(np.float32)
        matting = np.array(matting).astype(np.float32) / 255.0

        out_img = np.clip(matting * src_img + (1 - matting) * bg, 0, 255).astype(np.uint8)

        return Image.fromarray(out_img)

    def __call__(self, data):
        assert isinstance(data, dict)

        if 'matting' in data and 'image' in data:
            if isinstance(data['image'], Sequence) and isinstance(data['matting'], Sequence):
                data['image'] = [self.paste_matting(i, c) for i, c in zip(data['image'], data['matting'])]
            else:
                data['image'] = self.paste_matting(data['image'], data['matting'])

            if 'reference_image' in data and 'reference_matting' in data:
                data['reference_image'] = self.paste_matting(data['reference_image'], data['reference_matting'])
        else:
            raise Exception(f"Not found keys: [`image`, `condition`]")

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(bg_value: {self.bg_value})"


class BoxCrop(object):
    def __init__(self,
                 crop_size=(1024, 768),
                 pad_val=255,
                 pad_bbox=5,
                 prob=1.0):
        _log_api_usage_once(self)
        if isinstance(crop_size, int):
            crop_size = [crop_size, ] * 2
        elif isinstance(crop_size, Sequence):
            assert len(crop_size) == 2
        else:
            raise Exception(f"Error crop_size type: {type(crop_size)}")

        self.crop_size = crop_size
        self.pad_val = pad_val
        self.pad_bbox = pad_bbox
        self.prob = prob

    def center_crop(self, img, cond_img=None):
        _, h, w = F.get_dimensions(img)
        ch, cw = self.crop_size
        img = np.array(img)
        if cond_img is not None:
            cond_img = np.array(cond_img)

        if ch <= h and cw <= w:
            y1 = (h - ch) // 2
            x1 = (w - cw) // 2
            img = img[y1: y1 + ch, x1: x1 + cw]
            if cond_img is not None:
                cond_img = cond_img[y1: y1 + ch, x1: x1 + cw]
        else:
            ratio_h, ratio_w = ch / h, cw / w
            # 短边resize
            if ratio_h < ratio_w:
                oh = int(w / cw * ch)
                y1 = (h - oh) // 2
                img = img[y1: y1 + oh]
                if cond_img is not None:
                    cond_img = cond_img[y1: y1 + oh]
            else:
                ow = int(h / ch * cw)
                x1 = (w - ow) // 2
                img = img[:, x1: x1 + ow]
                if cond_img is not None:
                    cond_img = cond_img[:, x1: x1 + ow]

        if cond_img is not None:
            cond_img = Image.fromarray(cond_img).resize((cw, ch), 1)

        return Image.fromarray(img).resize((cw, ch), 1), cond_img

    def crop_and_resize(self, src_img, bboxes, cond_img=None):
        _, height, width = F.get_dimensions(src_img)
        bboxes *= np.array([width, height, width, height])

        if len(bboxes) == 0:
            return self.center_crop(src_img, cond_img)
        else:
            det_bbox = np.int32(bboxes[0])
            # area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
            # ind = np.argmax(area)
            # det_bbox = np.int32(bboxes[ind])

        crop_bbox, pad_ = crop_human_bbox(
            det_bbox, (height, width), self.crop_size, self.pad_bbox)
        x1, y1, x2, y2 = crop_bbox
        src_img = np.array(src_img)
        crop_img = src_img[y1: y2, x1: x2]
        if cond_img is not None:
            cond_img = np.array(cond_img)
            cond_img = cond_img[y1: y2, x1: x2]
        if pad_ is not None:
            crop_img = np.pad(crop_img, pad_, constant_values=self.pad_val)
            if cond_img is not None:
                cond_img = np.pad(cond_img, pad_, constant_values=0)

        crop_img = Image.fromarray(crop_img).resize(self.crop_size[::-1], Image.LANCZOS)
        if cond_img is not None:
            cond_img = Image.fromarray(cond_img).resize(self.crop_size[::-1], Image.LANCZOS)
        return crop_img, cond_img

    def __call__(self, data):
        assert isinstance(data, dict)

        if 'image' in data and 'condition' in data and random.random() <= self.prob:
            img, cond_img = self.crop_and_resize(
                data['image'],
                data['condition']['body']['bboxes'],
                cond_img=data['condition_image'] if 'condition_image' in data else None,
            )
            data['image'] = img
            if 'condition_image' in data:
                data['condition_image'] = cond_img

        if 'reference_image' in data and 'reference_condition' in data:
            img, _ = self.crop_and_resize(
                data['reference_image'], data['reference_condition']['body']['bboxes'])
            data['reference_image'] = img

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(crop_size: {self.crop_size}, pad_val: {self.pad_val})"


class HumanCrop(object):
    def __init__(self,
                 size,
                 prob=1.0,
                 pad_bbox=100,
                 pad_val=255,
                 is_max_area_bbox=False,
                 random_crop=False,
                 w_ratio=0.8,
                 h_ratio=0.75):
        _log_api_usage_once(self)
        if not isinstance(size, (int, Sequence)):
            raise TypeError(f"Size should be int or sequence. Got {type(size)}")
        if isinstance(size, Sequence) and len(size) not in (1, 2):
            raise ValueError("If size is a sequence, it should have 1 or 2 values")
        self.size = size
        self.prob = prob
        self.pad_bbox = pad_bbox
        self.pad_val = pad_val
        self.is_max_area_bbox = is_max_area_bbox
        self.random_crop = random_crop
        self.w_ratio = w_ratio
        self.h_ratio = h_ratio
        self.resize = Resize(size, interpolation=F.InterpolationMode.LANCZOS)

    def pad_image(self, img, pad_values=255):
        img = np.array(img)
        h, w = img.shape[:2]
        pad_border = None
        if w > h:
            pad_ = w - h
            pad_border = ((pad_ // 2, pad_ - pad_ // 2), (0, 0), (0, 0))
            img = np.pad(
                img,
                pad_border,
                constant_values=pad_values
            )
        elif h > w:
            pad_ = h - w
            pad_border = ((0, 0), (pad_ // 2, pad_ - pad_ // 2), (0, 0))
            img = np.pad(
                img,
                pad_border,
                constant_values=pad_values
            )
        return Image.fromarray(img)

    def get_params(self, src_size, output_size):
        h, w = src_size
        th, tw = output_size

        if h < th or w < tw:
            raise ValueError(f"Required crop size {(th, tw)} is larger than input image size {(h, w)}")

        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return [i, j, th, tw]

    def crop_human(self, src_img, bboxes, pad_bbox=0):
        if len(bboxes) == 0:
            return self.pad_image(src_img, pad_values=self.pad_val)
        else:
            if self.is_max_area_bbox and len(bboxes) > 1:
                area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
                ind = np.argmax(area)
                det_bbox = bboxes[ind]
            else:
                det_bbox = bboxes[0]

        _, height, width = F.get_dimensions(src_img)
        det_bbox *= np.array([width, height, width, height])
        det_bbox = np.int32(det_bbox)
        x1, y1, x2, y2 = det_bbox
        x1 = min(max(x1, 0), width)
        x2 = min(max(x2, 0), width)
        y1 = min(max(y1, 0), height)
        y2 = min(max(y2, 0), height)
        if pad_bbox > 0:
            x1 = x1 - pad_bbox if x1 - pad_bbox > 0 else 0
            y1 = y1 - pad_bbox if y1 - pad_bbox > 0 else 0
            x2 = x2 + pad_bbox if x2 + pad_bbox < width else width
            y2 = y2 + pad_bbox if y2 + pad_bbox < height else height

        if x2 - x1 < 0.1 * width or y2 - y1 < 0.1 * height:
            return self.pad_image(src_img, pad_values=self.pad_val)

        src_img = np.array(src_img)
        crop_img = src_img[y1: y2, x1: x2]
        # random crop
        if self.random_crop and random.random() <= 0.5:
            w, h = (x2 - x1), (y2 - y1)
            cw = random.randint(int(self.w_ratio * w), w)
            ch = random.randint(int(self.h_ratio * h), h)
            i, j, th, tw = self.get_params((h, w), (ch, cw))
            crop_img = crop_img[i: i + th, j: j + tw]
        # pad image to square
        crop_img = self.pad_image(crop_img, pad_values=self.pad_val)
        return crop_img

    def __call__(self, data):
        if not isinstance(data, dict):
            return data

        if 'image' in data and 'condition' in data and random.random() <= self.prob:
            pad_bbox = random.randint(0, self.pad_bbox)
            img = self.crop_human(
                data['image'],
                data['condition']['body']['bboxes'],
                pad_bbox=pad_bbox,
            )
            data['image'] = img
        else:
            data['image'] = self.pad_image(data['image'], pad_values=self.pad_val)

        data['image'] = self.resize(data['image'])
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(prob: {self.prob}, pad_bbox: {self.pad_bbox})"


class RandomSelects(object):
    def __init__(self, transforms):
        _log_api_usage_once(self)
        self.transforms = transforms

    def __call__(self, data):
        trans = random.choice(self.transforms)
        data = trans(data)

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"


class RandomRectMask(object):
    def __init__(self,
                 size,
                 w_ratio=0.45,
                 h_ratio=0.45,
                 max_ratio=0.95,
                 scale_factor=16):
        _log_api_usage_once(self)
        if not isinstance(size, (int, Sequence)):
            raise TypeError(f"Size should be int or sequence. Got {type(size)}")
        if isinstance(size, Sequence) and len(size) not in (1, 2):
            raise ValueError("If size is a sequence, it should have 1 or 2 values")
        self.size = size
        self.w_ratio = w_ratio
        self.h_ratio = h_ratio
        self.max_ratio = max_ratio
        self.scale_factor = scale_factor
        self.resize = Resize(size, interpolation=F.InterpolationMode.LANCZOS, max_size=1440)

    def scale_image(self, img):
        w, h = img.size
        w = int(w // self.scale_factor * self.scale_factor)
        h = int(h // self.scale_factor * self.scale_factor)
        img = np.array(img)[:h, :w]
        return Image.fromarray(img)

    def get_params(self, src_size, output_size):
        h, w = src_size
        th, tw = output_size

        if h < th or w < tw:
            raise ValueError(f"Required crop size {(th, tw)} is larger than input image size {(h, w)}")

        if w == tw and h == th:
            return [0, 0, h, w]

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return [i, j, th, tw]

    def get_rect_mask(self, img, rect_params):
        w, h = img.size
        i, j, th, tw = rect_params
        mask = torch.zeros((h, w), dtype=torch.float32)
        mask[i: i + th, j: j + tw] = 1
        mask = 1 - mask

        return mask

    def __call__(self, data):
        if isinstance(data, dict) and 'image' in data:
            img = data['image']
            img = self.resize(img)
            if self.scale_factor is not None:
                img = self.scale_image(img)
            W, H = img.size
            rect_w = random.randint(int(W * self.w_ratio), int(W * self.max_ratio))
            rect_h = random.randint(int(H * self.h_ratio), int(H * self.max_ratio))
            rect_params = self.get_params((H, W), (rect_h, rect_w))

            mask = self.get_rect_mask(img, rect_params)

            data['image'] = img
            data['mask'] = mask

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"


class RandomHumanMask(object):
    def __init__(self,
                 size,
                 max_ratio=0.9,
                 scale_factor=16):
        _log_api_usage_once(self)
        if not isinstance(size, (int, Sequence)):
            raise TypeError(f"Size should be int or sequence. Got {type(size)}")
        if isinstance(size, Sequence) and len(size) not in (1, 2):
            raise ValueError("If size is a sequence, it should have 1 or 2 values")
        self.size = size
        self.max_ratio = max_ratio
        self.scale_factor = scale_factor
        self.resize = Resize(size, interpolation=F.InterpolationMode.LANCZOS, max_size=1440)
        self.random_rect_mask = RandomRectMask(size, scale_factor=scale_factor)

    def scale_image(self, img, mask):
        w, h = img.size
        w = int(w // self.scale_factor * self.scale_factor)
        h = int(h // self.scale_factor * self.scale_factor)
        img = np.array(img)[:h, :w]
        mask = mask[:h, :w]
        return Image.fromarray(img), mask

    def get_rect_mask(self, img, rect_params):
        w, h = img.size
        i, j, th, tw = rect_params
        mask = torch.zeros((h, w), dtype=torch.float32)
        mask[i: i + th, j: j + tw] = 1
        mask = 1 - mask

        return mask

    def __call__(self, data):
        if isinstance(data, dict) and 'image' in data:
            if 'pose' in data:
                img = data['image']
                img = self.resize(img)
                w, h = img.size
                bboxes = data['pose']['body']['bboxes']
                if len(bboxes) > 0:
                    bbox = np.int32(bboxes[0] * np.array([w, h, w, h]))
                    bw = bbox[2] - bbox[0]
                    bh = bbox[3] - bbox[1]
                    if random.random() < 0.5:
                        x1 = random.randint(0, bbox[0])
                        y1 = random.randint(0, bbox[1])
                        x2 = random.randint(bbox[2], w)
                        cy = int(bbox[1] + bh / 2)
                        y2 = random.randint(cy, h)
                        rect_params = [y1, x1, y2 - y1, x2 - x1]
                    else:
                        rect_params = [bbox[1], bbox[0], bh, bw]

                    mask = self.get_rect_mask(img, rect_params)
                    if self.scale_factor is not None:
                        img, mask = self.scale_image(img, mask)

                    data['image'] = img
                    data['mask'] = mask

                    return data

            data = self.random_rect_mask(data)
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"


class PTRandomCrop(object):
    def __init__(self,
                 size,
                 prob=0.5,
                 condition='pose',
                 pad_square=False,
                 scale_factor=16):
        _log_api_usage_once(self)
        if not isinstance(size, (int, Sequence)):
            raise TypeError(f"Size should be int or sequence. Got {type(size)}")
        if isinstance(size, Sequence) and len(size) not in (1, 2):
            raise ValueError("If size is a sequence, it should have 1 or 2 values")

        self.size = size
        self.prob = prob
        self.condition = condition
        self.scale_factor = scale_factor
        self.pad_square = pad_square

    def scale_image(self, img, condition_img):
        w, h = img.size
        w = int(w // self.scale_factor * self.scale_factor)
        h = int(h // self.scale_factor * self.scale_factor)
        img = np.array(img)[:h, :w]
        condition_img = np.array(condition_img)[:h, :w]
        return Image.fromarray(img), Image.fromarray(condition_img)

    def pad_image(self, img, condition_img):
        w, h = img.size
        pad_border = [[0, 0], [0, 0], [0, 0]]
        # padding成1:1方图
        if w > h:
            pad_ = w - h
            pad_border = ((pad_ // 2, pad_ - pad_ // 2), (0, 0), (0, 0))
        elif h > w:
            pad_ = h - w
            pad_border = ((0, 0), (pad_ // 2, pad_ - pad_ // 2), (0, 0))

        img = np.pad(np.array(img), pad_border, constant_values=255)
        condition_img = np.pad(np.array(condition_img), pad_border,
                               constant_values=0 if self.condition == 'pose' else 255)

        return Image.fromarray(img), Image.fromarray(condition_img)

    def square_image_proc(self, img, condition_img):
        img, condition_img = self.pad_image(img, condition_img)
        img = img.resize((self.size, self.size), 1)
        condition_img = condition_img.resize(
            (self.size, self.size), 0 if self.condition == 'pose' else 1)
        return img, condition_img

    def center_crop(self, img, condition_img):
        _, image_height, image_width = F.get_dimensions(img)
        crop_height, crop_width = self.size

        if crop_width > image_width or crop_height > image_height:
            padding_ltrb = [
                (crop_width - image_width) // 2 if crop_width > image_width else 0,
                (crop_height - image_height) // 2 if crop_height > image_height else 0,
                (crop_width - image_width + 1) // 2 if crop_width > image_width else 0,
                (crop_height - image_height + 1) // 2 if crop_height > image_height else 0,
            ]
            img = F.pad(img, padding_ltrb, fill=255)
            condition_img = F.pad(
                condition_img, padding_ltrb, fill=0 if self.condition == 'pose' else 255)
            _, image_height, image_width = F.get_dimensions(img)
            if crop_width == image_width and crop_height == image_height:
                return img, condition_img

        crop_top = int(round((image_height - crop_height) / 2.0))
        crop_left = int(round((image_width - crop_width) / 2.0))
        img = F.crop(img, crop_top, crop_left, crop_height, crop_width)
        condition_img = F.crop(condition_img, crop_top, crop_left, crop_height, crop_width)
        return img, condition_img

    def long_side_resize(self, image, size, resample=1):
        w, h = image.size
        max_l = max(w, h)
        ratio = size / max_l
        new_w = size if w == max_l else int(w * ratio)
        new_h = size if h == max_l else int(h * ratio)
        return image.resize((new_w, new_h), resample)

    def other_image_proc(self, img, condition_img):
        if isinstance(self.size, Sequence):
            w, h = img.size
            oh = self.size[0]
            ow = int(oh / h * w) if oh != h else w
            img = img.resize((ow, oh), 1)
            condition_img = condition_img.resize(
                (ow, oh), 0 if self.condition == 'pose' else 1)

            img, condition_img = self.center_crop(img, condition_img)
        else:
            img = self.long_side_resize(img, self.size)
            condition_img = self.long_side_resize(
                condition_img, self.size, resample=0 if self.condition == 'pose' else 1)

        return img, condition_img

    def get_data(self, data):
        img = data['image']
        condition_img = data.get('condition_image', data['condition'])
        if self.pad_square:
            img, condition_img = self.square_image_proc(img, condition_img)
        else:
            img, condition_img = self.other_image_proc(img, condition_img)

        img, condition_img = self.scale_image(img, condition_img)

        data['image'] = img
        data['condition_image'] = condition_img
        return data

    def crop_data(self, data):
        _, height, width = F.get_dimensions(data['image'])
        det_bbox = data['condition']['body']['bboxes'][0]
        det_bbox *= np.array([width, height, width, height])
        det_bbox = np.int32(det_bbox)
        x1, y1, x2, y2 = det_bbox
        x1 = min(max(x1, 0), width)
        x2 = min(max(x2, 0), width)
        y1 = min(max(y1, 0), height)
        y2 = min(max(y2, 0), height)

        new_x1 = random.randint(0, x1)
        new_x2 = random.randint(x2, width)
        new_y1 = random.randint(0, y1)
        new_y2 = random.randint(y2, height)

        img = np.array(data['image'])[new_y1: new_y2, new_x1: new_x2]
        pose_img = np.array(data['condition_image'])[new_y1: new_y2, new_x1: new_x2]

        data['image'] = Image.fromarray(img)
        data['condition_image'] = Image.fromarray(pose_img)
        return data

    def __call__(self, data):
        if 'condition' in data and 'image' in data:
            if isinstance(data['condition'], dict):
                bboxes = data['condition']['body']['bboxes']
                if len(bboxes) > 0:
                    det_bbox = bboxes[0]
                    x1, y1, x2, y2 = det_bbox
                    if (x2 - x1 > 0.3 or y2 - y1 > 0.4) and random.random() < self.prob:
                        data = self.crop_data(data)

            data = self.get_data(data)

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"


class FLUXICFillProc(object):
    def __init__(self,
                 size=(1280, 960),
                 prob=0.5,
                 max_size=None,
                 scale_factor=16):
        _log_api_usage_once(self)
        assert isinstance(size, Sequence) and len(size) == 2

        self.size = size
        self.prob = prob
        self.scale_factor = scale_factor

    def center_crop(self, img):
        _, image_height, image_width = F.get_dimensions(img)
        crop_height, crop_width = self.size

        if crop_width > image_width or crop_height > image_height:
            padding_ltrb = [
                (crop_width - image_width) // 2 if crop_width > image_width else 0,
                (crop_height - image_height) // 2 if crop_height > image_height else 0,
                (crop_width - image_width + 1) // 2 if crop_width > image_width else 0,
                (crop_height - image_height + 1) // 2 if crop_height > image_height else 0,
            ]
            img = F.pad(img, padding_ltrb, fill=255)  # PIL uses fill value 0
            _, image_height, image_width = F.get_dimensions(img)
            if crop_width == image_width and crop_height == image_height:
                return img

        crop_top = int(round((image_height - crop_height) / 2.0))
        crop_left = int(round((image_width - crop_width) / 2.0))
        return F.crop(img, crop_top, crop_left, crop_height, crop_width)

    def get_image(self, img):
        w, h = img.size
        oh = self.size[0]
        ow = int(oh / h * w) if oh != h else w
        img = img.resize((ow, oh), 1)
        img = self.center_crop(img)
        return img

    def crop_data(self, data):
        _, height, width = F.get_dimensions(data['image'])
        det_bbox = data['condition']['body']['bboxes'][0]
        det_bbox *= np.array([width, height, width, height])
        det_bbox = np.int32(det_bbox)
        x1, y1, x2, y2 = det_bbox
        x1 = min(max(x1, 0), width)
        x2 = min(max(x2, 0), width)
        y1 = min(max(y1, 0), height)
        y2 = min(max(y2, 0), height)

        new_x1 = random.randint(0, x1)
        new_x2 = random.randint(x2, width)
        new_y1 = random.randint(0, y1)
        new_y2 = random.randint(y2, height)

        img = np.array(data['image'])[new_y1: new_y2, new_x1: new_x2]
        pose_img = np.array(data['condition_image'])[new_y1: new_y2, new_x1: new_x2]

        data['image'] = Image.fromarray(img)
        data['condition_image'] = Image.fromarray(pose_img)
        return data

    def __call__(self, data):
        if isinstance(data, dict):
            img = data['image']
            data['image'] = self.get_image(img)
        else:
            data = self.get_image(data)

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"


class TextPTRandomCrop(object):
    def __init__(self,
                 size,
                 crop_size=(1024, 768),
                 scale_factor=16):
        _log_api_usage_once(self)
        if not isinstance(size, (int, Sequence)):
            raise TypeError(f"Size should be int or sequence. Got {type(size)}")
        if isinstance(size, Sequence) and len(size) not in (1, 2):
            raise ValueError("If size is a sequence, it should have 1 or 2 values")

        self.size = size
        self.crop_size = crop_size
        self.scale_factor = scale_factor

    def scale_image(self, img):
        w, h = img.size
        w = int(w // self.scale_factor * self.scale_factor)
        h = int(h // self.scale_factor * self.scale_factor)
        img = np.array(img)[:h, :w]
        return Image.fromarray(img)

    def pad_image(self, img):
        w, h = img.size
        pad_border = [[0, 0], [0, 0], [0, 0]]
        # padding成1:1方图
        if w > h:
            pad_ = w - h
            pad_border = ((pad_ // 2, pad_ - pad_ // 2), (0, 0), (0, 0))
        elif h > w:
            pad_ = h - w
            pad_border = ((0, 0), (pad_ // 2, pad_ - pad_ // 2), (0, 0))

        img = np.pad(np.array(img), pad_border, constant_values=255)

        return Image.fromarray(img)

    def center_crop(self, img, size=(1024, 768)):
        _, image_height, image_width = F.get_dimensions(img)
        crop_height, crop_width = size

        if crop_width > image_width or crop_height > image_height:
            padding_ltrb = [
                (crop_width - image_width) // 2 if crop_width > image_width else 0,
                (crop_height - image_height) // 2 if crop_height > image_height else 0,
                (crop_width - image_width + 1) // 2 if crop_width > image_width else 0,
                (crop_height - image_height + 1) // 2 if crop_height > image_height else 0,
            ]
            img = F.pad(img, padding_ltrb, fill=255)
            _, image_height, image_width = F.get_dimensions(img)
            if crop_width == image_width and crop_height == image_height:
                return img

        crop_top = int(round((image_height - crop_height) / 2.0))
        crop_left = int(round((image_width - crop_width) / 2.0))
        img = F.crop(img, crop_top, crop_left, crop_height, crop_width)
        return img

    def long_side_resize(self, image, size=1024):
        w, h = image.size
        max_l = max(w, h)
        ratio = size / max_l
        new_w = size if w == max_l else int(w * ratio)
        new_h = size if h == max_l else int(h * ratio)
        return image.resize((new_w, new_h), 1)

    def crop_human_bbox(self, det_bbox, img_size):
        x1, y1, x2, y2 = det_bbox
        h, w = img_size
        ch, cw = self.crop_size

        bh, bw = y2 - y1, x2 - x1
        ratio_h, ratio_w = ch / bh, cw / bw

        # 长边resize
        if ratio_h < ratio_w:
            # 按高 resize
            ow = int(bh / ch * cw)
            expand_w = ow - bw
            offset = int(expand_w / 2)

            if x1 - offset >= 0 and x2 + (expand_w - offset) <= w:
                x1 -= offset
                x2 += (expand_w - offset)
            elif x1 - offset < 0:
                x1 = 0
                x2 = min(ow, w)
            else:
                x2 = w
                x1 = max(w - ow, 0)

            return [x1, y1, x2, y2], [bh, ow]
        elif ratio_h > ratio_w:
            # 按宽 resize
            oh = int(bw / cw * ch)
            expand_h = oh - bh
            offset = int(expand_h / 2)

            if y1 - offset >= 0 and y2 + (expand_h - offset) <= h:
                y1 -= offset
                y2 += (expand_h - offset)
            elif y1 - offset < 0:
                y1 = 0
                y2 = min(oh, h)
            else:
                y2 = h
                y1 = max(h - oh, 0)

            return [x1, y1, x2, y2], [oh, bw]
        else:
            return [x1, y1, x2, y2], [bh, bw]

    def crop_data(self, data, pad=20):
        _, height, width = F.get_dimensions(data['image'])
        det_bbox = data['condition']['body']['bboxes'][0]
        det_bbox = np.int32(det_bbox * np.array([width, height, width, height]))
        x1, y1, x2, y2 = det_bbox
        x1 = min(max(x1 - pad, 0), width)
        x2 = min(max(x2 + pad, 0), width)
        y1 = min(max(y1 - pad, 0), height)
        y2 = min(max(y2 + pad, 0), height)

        crop_bbox, out_size = self.crop_human_bbox([x1, y1, x2, y2], (height, width))
        x1, y1, x2, y2 = crop_bbox
        img = np.array(data['image'])[y1: y2, x1: x2]
        img = self.center_crop(Image.fromarray(img), size=out_size)
        img = img.resize(self.crop_size[::-1], 1)
        data['image'] =img
        return data

    def get_data(self, data):
        img = data['image']
        img = self.long_side_resize(img, self.size)
        img = self.scale_image(img)
        data['image'] = img
        return data

    def __call__(self, data):

        if data.get("crop_image", False):
            if len(data['condition']['body']['bboxes']) > 0:
                data = self.crop_data(data)
            else:
                img = data['image']
                img = self.long_side_resize(img, size=self.crop_size[0])
                data['image'] = self.center_crop(img, size=self.crop_size)
        else:
            data = self.get_data(data)

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"
