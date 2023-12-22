#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The VIP Inc. AIGC team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import os
from os.path import join, splitext
import numpy as np
import pandas
import random
from tqdm.auto import tqdm
from collections import OrderedDict
from PIL import Image
import torch
from torchvision import transforms as tv_transforms

from .transforms import *
from .utils import *
from diffusers.utils.vip_utils import load_image
from .videoreader import VideoReader


__all__ = [
    'T2IDataset', 'ControlNetDataset', 'FaceDataset',
    'ConDepthDataset', 'ConPoseDataset', 'ConCannyDataset',
    'AnimateAnyoneDataset',
]


class T2IDataset(torch.utils.data.Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        dataset_csv,
        train_data_dir,
        tokenizer,
        img_size=512,
        center_crop=False,
        random_hflip=False,
        random_vflip=False,
        drop_text=0.1,
        keep_in_memory=False
    ):
        assert os.path.exists(dataset_csv)
        assert os.path.exists(train_data_dir)
        if drop_text < 0. or drop_text > 1.:
            raise ValueError("`drop_text` must be in the range [0., 1.].")

        self.dataset_csv = dataset_csv
        self.train_data_dir = train_data_dir
        self.tokenizer = tokenizer
        self.drop_text = drop_text
        self.keep_in_memory = keep_in_memory
        self.empty_text_inputs = tokenizer(
            "", max_length=tokenizer.model_max_length,
            padding="max_length", truncation=True, return_tensors="pt")

        self.metadata = pandas.read_csv(dataset_csv).values.tolist()
        self._length = len(self.metadata)
        self.image_list = []
        if keep_in_memory:
            for name, caption in tqdm(self.metadata):
                img = load_image(join(train_data_dir, name))
                self.image_list.append(img)

        self.image_transforms = tv_transforms.Compose(
            [
                Resize(img_size, interpolation=tv_transforms.InterpolationMode.LANCZOS),
                CenterCrop(img_size) if center_crop else RandomCrop(img_size),
                RandomHorizontalFlip() if random_hflip else tv_transforms.Lambda(lambda x: x),
                RandomVerticalFlip() if random_vflip else tv_transforms.Lambda(lambda x: x),
                ToTensor(),
                Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        if self.keep_in_memory:
            example["pixel_values"] = self.image_transforms(self.image_list[index])
        else:
            name = self.metadata[index][0]
            img = load_image(join(self.train_data_dir, name))
            example["pixel_values"] = self.image_transforms(img)

        if random.random() < self.drop_text:
            text_inputs = self.empty_text_inputs
        else:
            captions = self.metadata[index][1]
            text_inputs = self.tokenizer(
                captions, max_length=self.tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt"
            )
        example["input_ids"] = text_inputs.input_ids

        return example


class ControlNetDataset(torch.utils.data.Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        dataset_csv,
        train_data_dir,
        condition_data_dir,
        tokenizer,
        condition_image=False,
        img_size=512,
        center_crop=False,
        random_flip=False,
        drop_text=0.1,
        keep_in_memory=False,
        data_transform="pose",
    ):
        assert os.path.exists(dataset_csv)
        assert os.path.exists(train_data_dir)
        if drop_text < 0. or drop_text > 1.:
            raise ValueError("`drop_text` must be in the range [0., 1.].")

        self.dataset_csv = dataset_csv
        self.train_data_dir = train_data_dir
        self.condition_data_dir = condition_data_dir
        self.tokenizer = tokenizer
        self.condition_image = condition_image
        self.center_crop = center_crop
        self.random_flip = random_flip
        self.drop_text = drop_text
        self.keep_in_memory = keep_in_memory
        self.empty_text_inputs = tokenizer(
            "", max_length=tokenizer.model_max_length,
            padding="max_length", truncation=True, return_tensors="pt")

        self.metadata = pandas.read_csv(dataset_csv).values.tolist()
        self._length = len(self.metadata)
        self.image_list = []
        if keep_in_memory:
            for name, caption in tqdm(self.metadata):
                data = {'image': load_image(join(train_data_dir, name))}
                if condition_image:
                    data['condition_image'] = load_image(join(condition_data_dir, name))
                else:
                    data['condition'] = pkl_load(
                        join(condition_data_dir, splitext(name)[0] + '_pose.pkl'))
                self.image_list.append(data)

        if data_transform == "pose":
            self.image_transforms = tv_transforms.Compose(
                [
                    Resize(img_size, interpolation=tv_transforms.InterpolationMode.LANCZOS),
                    DrawPose() if not self.condition_image else tv_transforms.Lambda(lambda x: x),
                    CenterCrop(img_size) if self.center_crop else RandomCrop(img_size),
                    RandomHorizontalFlip() if self.random_flip else tv_transforms.Lambda(lambda x: x),
                    ToTensor(),
                ]
            )
        elif data_transform == "color":
            self.image_transforms = tv_transforms.Compose(
                [
                    Resize(img_size, interpolation=tv_transforms.InterpolationMode.LANCZOS),
                    CenterCrop(img_size) if self.center_crop else RandomCrop(img_size),
                    RandomHorizontalFlip() if self.random_flip else tv_transforms.Lambda(lambda x: x),
                    ColorJitter(0.5, 0.5, 0.5, 0.5),
                    ToTensor(),
                ]
            )
        self.img_normalize = Normalize([0.5], [0.5])

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        if self.keep_in_memory:
             data = self.image_transforms(self.image_list[index])
             example["pixel_values"] = self.img_normalize(data['image'])
             example["conditioning_pixel_values"] = data['condition_image']
        else:
            name = self.metadata[index][0]
            data = {'image': load_image(join(self.train_data_dir, name))}
            if self.condition_image:
                data['condition_image'] = load_image(join(self.condition_data_dir, name))
            else:
                data['condition'] = pkl_load(
                    join(self.condition_data_dir, splitext(name)[0] + '_pose.pkl'))
            data = self.image_transforms(data)
            example["pixel_values"] = self.img_normalize(data['image'])
            example["conditioning_pixel_values"] = data['condition_image']

        if random.random() < self.drop_text:
            text_inputs = self.empty_text_inputs
        else:
            captions = self.metadata[index][1]
            text_inputs = self.tokenizer(
                captions, max_length=self.tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt"
            )
        example["input_ids"] = text_inputs.input_ids

        return example


class FaceDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            dataset_csv,
            train_data_dir,
            tokenizer,
            img_size=512,
            drop_text=0.1,
            keep_in_memory=False
    ):
        assert os.path.exists(dataset_csv)
        assert os.path.exists(train_data_dir)
        if drop_text < 0. or drop_text > 1.:
            raise ValueError("`drop_text` must be in the range [0., 1.].")

        self.dataset_csv = dataset_csv
        self.train_data_dir = train_data_dir
        self.tokenizer = tokenizer
        self.drop_text = drop_text
        self.keep_in_memory = keep_in_memory
        self.empty_text_inputs = tokenizer(
            "", max_length=tokenizer.model_max_length,
            padding="max_length", truncation=True, return_tensors="pt")

        self.metadata = pandas.read_csv(dataset_csv).values.tolist()
        self._length = len(self.metadata)
        if keep_in_memory:
            self.image_list = []
            for name, caption in tqdm(self.metadata):
                img = load_image(join(train_data_dir, name))
                self.image_list.append(img)

        self.image_transforms = tv_transforms.Compose(
            [
                Resize(img_size, interpolation=tv_transforms.InterpolationMode.LANCZOS),
                RandomCrop(img_size),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        name, captions = self.metadata[index]

        example = {}
        if self.keep_in_memory:
            example["pixel_values"] = self.image_transforms(self.image_list[index])
        else:
            image = load_image(join(self.train_data_dir, name))
            example["pixel_values"] = self.image_transforms(image)

        if random.random() < self.drop_text:
            text_inputs = self.empty_text_inputs
        else:
            text_inputs = self.tokenizer(
                captions, max_length=self.tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt"
            )
        example["input_ids"] = text_inputs.input_ids

        return example


class ConDepthDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_csv,
        train_data_dir,
        condition_data_dir,
        tokenizer,
        img_size=512,
        drop_text=0.1
    ):
        assert os.path.exists(dataset_csv)
        assert os.path.exists(train_data_dir)
        assert os.path.exists(condition_data_dir)
        if drop_text < 0. or drop_text > 1.:
            raise ValueError("`drop_text` must be in the range [0., 1.].")

        self.dataset_csv = dataset_csv
        self.train_data_dir = train_data_dir
        self.condition_data_dir = condition_data_dir
        self.tokenizer = tokenizer
        self.drop_text = drop_text
        self.empty_text_inputs = tokenizer(
            "", max_length=tokenizer.model_max_length,
            padding="max_length", truncation=True, return_tensors="pt")

        self.metadata = pandas.read_csv(dataset_csv).values.tolist()
        self._length = len(self.metadata)

        self.image_transforms = tv_transforms.Compose(
            [
                Resize(img_size, interpolation=tv_transforms.InterpolationMode.LANCZOS),
                RandomCrop(img_size),
                RandomHorizontalFlip(),
                ToTensor(),
            ]
        )
        self.img_normalize = Normalize([0.5], [0.5])

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}

        name = self.metadata[index][0]
        data = {
            'image': load_image(join(self.train_data_dir, name)),
            'condition_image': load_image(join(self.condition_data_dir, splitext(name)[0] + '.jpg'))
        }
        data = self.image_transforms(data)
        example["pixel_values"] = self.img_normalize(data['image'])
        example["conditioning_pixel_values"] = data['condition_image']

        if random.random() < self.drop_text:
            text_inputs = self.empty_text_inputs
        else:
            captions = self.metadata[index][1]
            text_inputs = self.tokenizer(
                captions, max_length=self.tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt"
            )
        example["input_ids"] = text_inputs.input_ids

        return example


class ConPoseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_csv,
        train_data_dir,
        condition_data_dir,
        tokenizer,
        img_size=512,
        drop_text=0.1
    ):
        assert os.path.exists(dataset_csv)
        assert os.path.exists(train_data_dir)
        assert os.path.exists(condition_data_dir)
        if drop_text < 0. or drop_text > 1.:
            raise ValueError("`drop_text` must be in the range [0., 1.].")

        self.dataset_csv = dataset_csv
        self.train_data_dir = train_data_dir
        self.condition_data_dir = condition_data_dir
        self.tokenizer = tokenizer
        self.drop_text = drop_text
        self.empty_text_inputs = tokenizer(
            "", max_length=tokenizer.model_max_length,
            padding="max_length", truncation=True, return_tensors="pt")

        self.metadata = pandas.read_csv(dataset_csv).values.tolist()
        self._length = len(self.metadata)

        self.image_transforms = tv_transforms.Compose(
            [
                Resize(img_size, interpolation=tv_transforms.InterpolationMode.LANCZOS),
                DrawPose(),
                RandomCrop(img_size),
                RandomHorizontalFlip(),
                ToTensor(),
            ]
        )
        self.img_normalize = Normalize([0.5], [0.5])

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        name = self.metadata[index][0]
        data = {
            'image': load_image(join(self.train_data_dir, name)),
            'condition': pkl_load(
                join(self.condition_data_dir, splitext(name)[0] + '_pose.pkl'))
        }
        data = self.image_transforms(data)
        example["pixel_values"] = self.img_normalize(data['image'])
        example["conditioning_pixel_values"] = data['condition_image']

        if random.random() < self.drop_text:
            text_inputs = self.empty_text_inputs
        else:
            captions = self.metadata[index][1]
            text_inputs = self.tokenizer(
                captions, max_length=self.tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt"
            )
        example["input_ids"] = text_inputs.input_ids

        return example


class ConCannyDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_csv,
        train_data_dir,
        tokenizer,
        img_size=512,
        drop_text=0.1
    ):
        assert os.path.exists(dataset_csv)
        assert os.path.exists(train_data_dir)
        if drop_text < 0. or drop_text > 1.:
            raise ValueError("`drop_text` must be in the range [0., 1.].")

        self.dataset_csv = dataset_csv
        self.train_data_dir = train_data_dir
        self.tokenizer = tokenizer
        self.drop_text = drop_text
        self.empty_text_inputs = tokenizer(
            "", max_length=tokenizer.model_max_length,
            padding="max_length", truncation=True, return_tensors="pt")

        self.metadata = pandas.read_csv(dataset_csv).values.tolist()
        self._length = len(self.metadata)

        self.image_transforms = tv_transforms.Compose(
            [
                Resize(img_size, interpolation=tv_transforms.InterpolationMode.LANCZOS),
                RandomCrop(img_size),
                DrawCanny(),
                RandomHorizontalFlip(),
                ToTensor(),
            ]
        )
        self.img_normalize = Normalize([0.5], [0.5])

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        name = self.metadata[index][0]
        data = {'image': load_image(join(self.train_data_dir, name))}
        data = self.image_transforms(data)
        example["pixel_values"] = self.img_normalize(data['image'])
        example["conditioning_pixel_values"] = data['condition_image']

        if random.random() < self.drop_text:
            text_inputs = self.empty_text_inputs
        else:
            captions = self.metadata[index][1]
            text_inputs = self.tokenizer(
                captions, max_length=self.tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt"
            )
        example["input_ids"] = text_inputs.input_ids

        return example


class AnimateAnyoneDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            dataset_csv,
            train_data_dir,
            condition_data_dir,
            tokenizer,
            clip_processor=None,
            img_size=512,
            drop_text=0.1,
            num_frames=24,
            stride=4,
            is_video=False,
    ):
        assert os.path.exists(dataset_csv)
        assert os.path.exists(train_data_dir)
        assert os.path.exists(condition_data_dir)
        if drop_text < 0. or drop_text > 1.:
            raise ValueError("`drop_text` must be in the range [0., 1.].")

        self.dataset_csv = dataset_csv
        self.train_data_dir = train_data_dir
        self.condition_data_dir = condition_data_dir
        self.tokenizer = tokenizer
        self.clip_processor = clip_processor
        self.drop_text = drop_text
        self.num_frames = num_frames
        self.stride = stride
        self.is_video = is_video
        self.clip_length = (self.num_frames - 1) * self.stride + 1
        self.empty_text_inputs = tokenizer(
            "", max_length=tokenizer.model_max_length,
            padding="max_length", truncation=True, return_tensors="pt")

        self.metadata = pandas.read_csv(dataset_csv).values.tolist()

        self.video_to_image = OrderedDict()
        for name, captions in self.metadata:
            video_name = name.split('_')[0]
            if video_name not in self.video_to_image:
                self.video_to_image[video_name] = [name]
            else:
                self.video_to_image[video_name].append(name)

        self._length = len(self.metadata) if not is_video else len(self.video_to_image)
        self.video_names = list(self.video_to_image.keys())

        self.image_transforms = tv_transforms.Compose(
            [
                ResizePadToTensor(img_size, interpolation='bilinear'),
                RandomHorizontalFlip(),
            ]
        )
        self.img_normalize = Normalize([0.5], [0.5], inplace=True)

    def __len__(self):
        return self._length

    def to_tensor(self, img, dtype=torch.float32):
        img = np.array(img).transpose((2, 0, 1))
        img = torch.from_numpy(img).contiguous()

        return img.to(dtype).div(255.0)

    def get_frames(self, video_list):
        st_idx = random.randint(0, self.stride - 1)
        samples_idx = list(range(st_idx, len(video_list), self.stride))
        if len(samples_idx) >= self.num_frames:
            st_idx = random.randint(0, len(samples_idx) - self.num_frames)
            samples_idx = samples_idx[st_idx: st_idx + self.num_frames]
        else:
            samples_idx = samples_idx + [samples_idx[-1], ] * (self.num_frames - len(samples_idx))

        images, condition_images = [], []
        for i in samples_idx:
            name = video_list[i]
            img = load_image(join(self.train_data_dir, name))
            condition_img = load_image(join(self.condition_data_dir, name))

            images.append(self.to_tensor(img))
            condition_images.append(self.to_tensor(condition_img))

        images = torch.stack(images)
        condition_images = torch.stack(condition_images)
        ref_name = video_list[random.choice(samples_idx)]
        reference_image = load_image(join(self.train_data_dir, ref_name))

        return images, condition_images, reference_image

    def get_data(self, index):
        if self.is_video:
            # train with video
            captions = ""
            video_name = self.video_names[index]
            images, condition_images, reference_image = self.get_frames(
                self.video_to_image[video_name])
            data = {
                'image': images,
                'condition_image': condition_images,
                'reference_image': reference_image
            }
        else:
            # train with image
            name, captions = self.metadata[index]
            video_name = name.split('_')[0]
            video_list = self.video_to_image[video_name]
            if len(video_list) > 2 * self.clip_length:
                idx = video_list.index(name)
                st_idx = idx - self.clip_length if idx - self.clip_length >= 0 else 0
                end_idx = idx + self.clip_length if idx + self.clip_length < len(video_list) else len(video_list)

                st_idx = len(video_list) - 2 * self.clip_length if end_idx == len(video_list) else st_idx
                end_idx = 2 * self.clip_length if st_idx == 0 else end_idx
                ref_name = random.choice(video_list[st_idx: end_idx])
            else:
                ref_name = random.choice(video_list)
            data = {
                'image': self.to_tensor(load_image(join(self.train_data_dir, name))),
                'condition_image': self.to_tensor(load_image(join(self.condition_data_dir, name))),
                'reference_image': load_image(join(self.train_data_dir, ref_name))
            }
        data = self.image_transforms(data)

        return data, captions

    def __getitem__(self, index):
        example = {}
        data, captions = self.get_data(index)

        example["pixel_values"] = self.img_normalize(data['image'])
        example["conditioning_pixel_values"] = data['condition_image']

        if random.random() < self.drop_text:
            text_inputs = self.empty_text_inputs
            data['reference_image'] = Image.new("RGB", data['reference_image'].size)
        else:
            text_inputs = self.tokenizer(
                captions, max_length=self.tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt"
            )

        example["reference_pixel_values"] = self.img_normalize(ToTensor()(data['reference_image']))
        example["input_ids"] = text_inputs.input_ids
        if self.clip_processor is not None:
            example["reference_image"] = self.clip_processor(
                images=data["reference_image"], return_tensors="pt").pixel_values

        return example
