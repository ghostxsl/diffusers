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
from .vos_uploader import VOSUploader


__all__ = [
    'T2IDataset', 'ControlNetDataset', 'FaceDataset',
    'ConDepthDataset', 'ConPoseDataset', 'ConCannyDataset',
    'AnimateDataset', 'AnimateVosDataset'
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
        if drop_text < 0. or drop_text > 1.:
            raise ValueError("`drop_text` must be in the range [0., 1.].")

        self.tokenizer = tokenizer
        self.drop_text = drop_text
        self.empty_text_inputs = tokenizer(
            "", max_length=tokenizer.model_max_length,
            padding="max_length", truncation=True, return_tensors="pt")

        self.metadata = self.get_metadata(dataset_csv, train_data_dir, condition_data_dir)
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

    def get_metadata(self, dataset_csv, train_data_dir, condition_data_dir):
        print("Loading dataset...")
        dataset_csv = dataset_csv.split(',')
        train_data_dir = train_data_dir.split(',')
        condition_data_dir = condition_data_dir.split(',')

        out = []
        for csv_file, tdir, cdir in zip(dataset_csv, train_data_dir, condition_data_dir):
            temp = pandas.read_csv(csv_file).values.tolist()
            for name, caption in tqdm(temp):
                out.append([join(tdir, name), join(cdir, splitext(name)[0] + '.pose'), caption])
        return out

    def __getitem__(self, index):
        example = {}
        img_path, cond_path, caption = self.metadata[index]
        data = {
            'image': load_image(img_path),
            'condition': pkl_load(cond_path),
        }
        data = self.image_transforms(data)
        example["pixel_values"] = self.img_normalize(data['image'])
        example["conditioning_pixel_values"] = data['condition_image']

        if random.random() < self.drop_text:
            text_inputs = self.empty_text_inputs
        else:
            text_inputs = self.tokenizer(
                caption, max_length=self.tokenizer.model_max_length,
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


class AnimateDataset(torch.utils.data.Dataset):
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
            sample_stride=16,
            is_video=False,
            caption=""
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
        self.sample_stride = sample_stride
        self.is_video = is_video
        self.caption = caption
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

        self._length = len(self.metadata)
        if is_video:
            self.video_list = self.get_frames_name_list()
            self._length = len(self.video_list)

        self._draw = DrawPose(prob_hand=1.0, prob_face=1.0)
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
        def _to_tensor(img):
            img = np.array(img).transpose((2, 0, 1))
            img = torch.from_numpy(img).contiguous()
            return img.to(dtype).div(255.0)

        if isinstance(img, (list, tuple)):
            img = [_to_tensor(p) for p in img]
            img = torch.stack(img)
        else:
            img = _to_tensor(img)

        return img

    def get_frames_name_list(self):
        video_list = []
        for vname, vlist in self.video_to_image.items():
            vlist = np.array(vlist)
            samples_idx = np.array(list(range(0, len(vlist), self.stride)))
            vlist = vlist[samples_idx]
            num_samples = int(np.ceil((len(vlist) - self.num_frames) / self.sample_stride + 1))

            start_idx = 0
            for i in range(num_samples):
                if start_idx + self.num_frames > len(vlist):
                    video_list.append(
                        [vlist[j] for j in range(len(vlist) - self.num_frames, len(vlist))])
                else:
                    video_list.append(
                        [vlist[j] for j in range(start_idx, start_idx + self.num_frames)])
                start_idx += self.sample_stride
        return video_list

    def get_frames(self, video_list, random_sample=False):
        if random_sample:
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
                points = pkl_load(join(self.condition_data_dir, splitext(name)[0] + '.pose'))
                condition_img = self._draw.draw_pose(img, points)

                images.append(self.to_tensor(img))
                condition_images.append(self.to_tensor(condition_img))
            ref_name = video_list[random.choice(samples_idx)]
        else:
            images, condition_images = [], []
            for name in video_list:
                img = load_image(join(self.train_data_dir, name))
                points = pkl_load(join(self.condition_data_dir, splitext(name)[0] + '.pose'))
                condition_img = self._draw.draw_pose(img, points)

                images.append(img)
                condition_images.append(condition_img)
            ref_name = random.choice(video_list)

        reference_image = load_image(join(self.train_data_dir, ref_name))

        return images, condition_images, reference_image

    def get_data(self, index):
        if self.is_video:
            # train with video
            captions = self.caption
            vlist = self.video_list[index]
            images, condition_images, reference_image = self.get_frames(vlist)
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

            img = load_image(join(self.train_data_dir, name))
            points = pkl_load(join(self.condition_data_dir, splitext(name)[0] + '.pose'))
            data = {
                'image': img,
                'condition_image': self._draw.draw_pose(img, points),
                'reference_image': load_image(join(self.train_data_dir, ref_name))
            }
        data = self.image_transforms(data)
        data['image'] = self.to_tensor(data['image'])
        data['condition_image'] = self.to_tensor(data['condition_image'])

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


class AnimateVosDataset(torch.utils.data.Dataset):
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
            sample_stride=16,
            is_video=False,
            caption="",
            base_img_vos_path="http://gd17-ai-inner-storegw.api.vip.com/llm-cv-public/vid_gen/data/animate_anyone/frames_img_1227"
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
        self.sample_stride = sample_stride
        self.is_video = is_video
        self.caption = caption
        self.clip_length = (self.num_frames - 1) * self.stride + 1
        self.empty_text_inputs = tokenizer(
            "", max_length=tokenizer.model_max_length,
            padding="max_length", truncation=True, return_tensors="pt")

        self.metadata = pandas.read_csv(dataset_csv).values.tolist()

        self.video_to_image = OrderedDict()
        for name, captions in self.metadata:
            video_name = name.split('-')[0]
            if video_name not in self.video_to_image:
                self.video_to_image[video_name] = [name]
            else:
                self.video_to_image[video_name].append(name)

        self._length = len(self.metadata)
        if is_video:
            self.video_list = self.get_frames_name_list()
            self._length = len(self.video_list)

        self._draw = DrawPose(prob_hand=1.0, prob_face=1.0)
        self.image_transforms = tv_transforms.Compose(
            [
                ResizePadToTensor(img_size, interpolation='bilinear'),
                RandomHorizontalFlip(),
            ]
        )
        self.img_normalize = Normalize([0.5], [0.5], inplace=True)

        self.vos_obj = VOSUploader()
        self.base_img_vos_path = base_img_vos_path

    def __len__(self):
        return self._length

    def to_tensor(self, img, dtype=torch.float32):
        def _to_tensor(img):
            img = np.array(img).transpose((2, 0, 1))
            img = torch.from_numpy(img).contiguous()
            return img.to(dtype).div(255.0)

        if isinstance(img, (list, tuple)):
            img = [_to_tensor(p) for p in img]
            img = torch.stack(img)
        else:
            img = _to_tensor(img)

        return img

    # to get the image or pose vos url from the name
    def get_vos_url(self, name, is_pose):
        if not is_pose:
            vid_id, img_id = name.split('-')
            url = os.path.join(self.base_img_vos_path, vid_id, img_id)
        else:
            vid_id = name.split('-')[0]
            pose_id = name.split('-')[-1].replace('.jpg', '.pose')
            pose_bath_path = self.base_img_vos_path.replace('frames_img_1227', 'frames_pose_1227')
            url = os.path.join(pose_bath_path, vid_id, pose_id)
        return url

    def get_frames_name_list(self):
        video_list = []
        for vname, vlist in self.video_to_image.items():
            vlist = np.array(vlist)
            samples_idx = np.array(list(range(0, len(vlist), self.stride)))
            vlist = vlist[samples_idx]
            num_samples = int(np.ceil((len(vlist) - self.num_frames) / self.sample_stride + 1))

            start_idx = 0
            for i in range(num_samples):
                if start_idx + self.num_frames > len(vlist):
                    video_list.append(
                        [vlist[j] for j in range(len(vlist) - self.num_frames, len(vlist))])
                else:
                    video_list.append(
                        [vlist[j] for j in range(start_idx, start_idx + self.num_frames)])
                start_idx += self.sample_stride
        return video_list

    def get_frames(self, video_list, random_sample=False):
        if random_sample:
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
                points = pkl_load(join(self.condition_data_dir, splitext(name)[0] + '.pose'))
                condition_img = self._draw.draw_pose(img, points)

                images.append(self.to_tensor(img))
                condition_images.append(self.to_tensor(condition_img))
            ref_name = video_list[random.choice(samples_idx)]
        else:
            images, condition_images = [], []
            for name in video_list:
                img_url = self.get_vos_url(name, is_pose=False)
                img = self.vos_obj.display_vos(img_url)
                img = load_image(img)

                pose_url = self.get_vos_url(name, is_pose=True)
                points = self.vos_obj.get_keypoints_json(pose_url)
                condition_img = self._draw.draw_pose(img, points)
                condition_img = load_image(condition_img)

                images.append(img)
                condition_images.append(condition_img)
            ref_name = random.choice(video_list)

        ref_url = self.get_vos_url(ref_name, is_pose=False)
        reference_img = self.vos_obj.display_vos(ref_url)
        reference_img = load_image(reference_img)

        return images, condition_images, reference_img

    def get_data(self, index):
        if self.is_video:
            # train with video
            captions = self.caption
            vlist = self.video_list[index]
            images, condition_images, reference_img = self.get_frames(vlist)
            data = {
                'image': images,
                'condition_image': condition_images,
                'reference_image': reference_img
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

            img_url = self.get_vos_url(name, is_pose=False)
            img = self.vos_obj.display_vos(img_url)
            img = load_image(img)

            pose_url = self.get_vos_url(name, is_pose=True)
            points = self.vos_obj.get_keypoints_json(pose_url)
            condition_img = self._draw.draw_pose(img, points)
            condition_img = load_image(condition_img)

            ref_url = self.get_vos_url(ref_name, is_pose=False)
            reference_img = self.vos_obj.display_vos(ref_url)
            reference_img = load_image(reference_img)

            data = {
                'image': img,
                'condition_image': condition_img,
                'reference_image': reference_img
            }
        data = self.image_transforms(data)
        data['image'] = self.to_tensor(data['image'])
        data['condition_image'] = self.to_tensor(data['condition_image'])

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
