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
from os.path import join, splitext, split
from collections.abc import Sequence
import numpy as np
import pandas
import random
from tqdm.auto import tqdm
import copy
import decord
from PIL import Image
import torch
from torchvision import transforms as tv_transforms

from .transforms import *
from .utils import *
from diffusers.utils.vip_utils import load_image
from .vos_client import VOSClient


Image.MAX_IMAGE_PIXELS = None
decord.bridge.set_bridge("torch")


__all__ = [
    'T2IDataset', 'ControlNetDataset', 'FaceDataset',
    'ConDepthDataset', 'ConPoseDataset', 'ConCannyDataset',
    'AnimateDataset', 'PoseTransDataset', 'ImageVariationDataset', 'GroupImageVariationDataset',
    'ConXSPoseDataset', 'IPAPoseTransDataset', 'KolorsIPAPoseTransDataset',
    'FluxT2IDataset', 'FluxPTDataset',
    'FLUXICImageDataset', 'FluxFillDataset', 'FluxAnyPTDataset',
    'FluxFillICT2IDataset', 'FluxTextPTDataset',
    'WanFLF2VDataset', 'FluxKontextDataset',
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
    ):
        if drop_text < 0. or drop_text > 1.:
            raise ValueError("`drop_text` must be in the range [0., 1.].")

        self.dataset_csv = dataset_csv
        self.train_data_dir = train_data_dir
        self.tokenizer = tokenizer
        self.drop_text = drop_text
        self.empty_text_inputs = tokenizer(
            "", max_length=tokenizer.model_max_length,
            padding="max_length", truncation=True, return_tensors="pt")

        self.metadata = self.get_metadata(dataset_csv, train_data_dir)
        self._length = len(self.metadata)

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

    def get_metadata(self, dataset_csv, train_data_dir):
        print("Loading dataset...")
        dataset_csv = dataset_csv.split(',')
        train_data_dir = train_data_dir.split(',')

        out = []
        for csv_file, tdir in zip(dataset_csv, train_data_dir):
            temp = pandas.read_csv(csv_file.strip()).values.tolist()
            for name, caption in tqdm(temp):
                out.append([join(tdir.strip(), name), caption])
        return out

    def __getitem__(self, index):
        example = {}
        img_path, captions = self.metadata[index]
        img = load_image(img_path)
        example["pixel_values"] = self.image_transforms(img)

        if random.random() < self.drop_text:
            text_inputs = self.empty_text_inputs
        else:
            text_inputs = self.tokenizer(
                captions, max_length=self.tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt"
            )
        example["input_ids"] = text_inputs.input_ids

        return example


class ControlNetDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_file,
        tokenizer_clip=None,
        tokenizer_t5=None,
        clip_processor=None,
        img_size=512,
        prob_uncond=0.1,
        max_sequence_length=512,
        vos_bucket='public',
    ):
        self.dataset_file = dataset_file
        self.tokenizer_clip = tokenizer_clip
        self.tokenizer_t5 = tokenizer_t5
        self.clip_processor = clip_processor
        self.prob_uncond = prob_uncond
        self.max_sequence_length = max_sequence_length

        if self.tokenizer_t5 is not None:
            self.empty_clip = tokenizer_clip(
                "",
                padding="max_length",
                max_length=77,
                truncation=True,
                return_overflowing_tokens=False,
                return_length=False,
                return_tensors="pt",
            ).input_ids
            self.empty_t5 = tokenizer_t5(
                "",
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                return_length=False,
                return_overflowing_tokens=False,
                return_tensors="pt",
            ).input_ids

        self.metadata = self.get_metadata(dataset_file)
        self._length = len(self.metadata)
        self.vos = VOSClient(vos_bucket)

        self.image_transforms = tv_transforms.Compose([
            DrawPose(img_size, random_ratio=(0.75, 1.25)),
            RandomSelects([
                tv_transforms.Compose([
                    RandomResize(
                        img_size,
                        [int(img_size * 0.75), int(img_size * 1.25)],
                        prob=0.5,
                        interpolation=tv_transforms.InterpolationMode.LANCZOS),
                    RandomCrop(img_size, fill=255, pad_if_needed=True),
                ]),
                ResizePad(img_size),
            ]),
        ])
        self.normalize_ = tv_transforms.Compose([
            ToTensor(),
            Normalize([0.5], [0.5], inplace=True),
        ])

    def __len__(self):
        return self._length

    def get_metadata(self, dataset_file):
        print("Loading dataset...")
        dataset_file = [a.strip() for a in dataset_file.split(',') if len(a.strip()) > 0]

        out = []
        for file_ in dataset_file:
            temp_list = load_file(file_)
            if isinstance(temp_list, list):
                out += temp_list
            elif isinstance(temp_list, dict):
                for v in temp_list.values():
                    out += v
            else:
                raise Exception(f"Error dataset_file: ({type(temp_list)}){file_}")

        return out

    def get_data(self, index):
        item = self.metadata[index]
        img, pose = item["image"], item["pose"]
        img = self.vos.download_vos_pil(img)
        pose = self.vos.download_vos_pkl(pose)

        caption = item.get("caption", "")
        if isinstance(caption, (tuple, list)):
            caption = random.choice(caption)

        return {
            "image": load_image(img),
            "condition": pose,
        }, caption

    def __getitem__(self, index):
        example = {}
        try:
            data, caption = self.get_data(index)
        except:
            item = self.metadata[index]
            image_name = item["image"]
            print(f"error read: {image_name}.")
            index = random.randint(0, self._length - 1)
            data, caption = self.get_data(index)
        data = self.image_transforms(data)

        example["pixel_values"] = self.normalize_(data['image'])
        example["conditioning_pixel_values"] = self.normalize_(data['condition_image'])

        if self.clip_processor is not None:
            # prepare ip adapter input
            example["reference_image"] = self.clip_processor(
                images=data['image'], return_tensors="pt").pixel_values

        if random.random() < self.prob_uncond:
            example["uncond"] = torch.tensor([[0.]])
            if self.tokenizer_t5 is not None:
                example["input_ids"] = self.empty_t5
                example["pooled_input_ids"] = self.empty_clip
        else:
            example["uncond"] = torch.tensor([[1.]])
            if self.tokenizer_t5 is not None:
                example["input_ids"] = self.tokenizer_t5(
                    caption,
                    padding="max_length",
                    max_length=self.max_sequence_length,
                    truncation=True,
                    return_length=False,
                    return_overflowing_tokens=False,
                    return_tensors="pt",
                ).input_ids
                example["pooled_input_ids"] = self.tokenizer_clip(
                    caption,
                    padding="max_length",
                    max_length=77,
                    truncation=True,
                    return_length=False,
                    return_overflowing_tokens=False,
                    return_tensors="pt",
                ).input_ids

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
            temp_list = pandas.read_csv(csv_file.strip()).values.tolist()
            for name, caption in tqdm(temp_list):
                out.append([join(tdir.strip(), name), join(cdir.strip(), splitext(name)[0] + '.pose'), caption])
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
            dataset_file,
            train_data_dir,
            condition_data_dir,
            tokenizer,
            matting_data_dir=None,
            use_matting=False,
            clip_processor=None,
            img_size=512,
            drop_text=0.1,
            num_frames=24,
            stride=4,
            overlap_frame=8,
            is_video=False,
            caption="",
            use_vos=False,
    ):
        if drop_text < 0. or drop_text > 1.:
            raise ValueError("`drop_text` must be in the range [0., 1.].")

        self.train_data_dir = train_data_dir
        self.condition_data_dir = condition_data_dir
        self.tokenizer = tokenizer
        self.matting_data_dir = matting_data_dir
        self.use_matting = use_matting
        self.clip_processor = clip_processor
        self.drop_text = drop_text
        self.num_frames = num_frames
        self.stride = stride
        self.sample_stride = num_frames - overlap_frame
        self.is_video = is_video
        self.caption = caption
        self.use_vos = use_vos
        self.clip_length = (self.num_frames - 1) * self.stride + 1
        self.empty_text_inputs = tokenizer(
            "", max_length=tokenizer.model_max_length,
            padding="max_length", truncation=True, return_tensors="pt")

        self.metadata = self.get_metadata(dataset_file)

        if is_video:
            self.video_list = self.get_frames_name_list(self.metadata)
            self._length = len(self.video_list)
        else:
            self.item_list = []
            for k, items in tqdm(self.metadata.items()):
                self.item_list.extend([(k, item) for item in items])
            self._length = len(self.item_list)

        if use_vos:
            self.vos = VOSClient()

        self.image_transforms = tv_transforms.Compose(
            [
                DrawPose(),
                PasteMatting() if use_matting else tv_transforms.Lambda(lambda x: x),
                ResizePad(img_size, padding=isinstance(img_size, int)),
                RandomHorizontalFlip(),
            ]
        )
        self.img_normalize = Normalize([0.5], [0.5], inplace=True)

    def __len__(self):
        return self._length

    def get_metadata(self, dataset_file):
        print("Loading dataset...")
        if self.use_vos:
            dataset_file = [d.strip() for d in dataset_file.split(',') if len(d.strip()) > 0]
            out = {}
            for file_path in dataset_file:
                temp = load_file(file_path)
                out.update(temp)
        else:
            out = load_file(dataset_file)

        return out

    def to_tensor(self, img, dtype=torch.float32):
        def _to_tensor(img):
            img = np.array(img).transpose((2, 0, 1))
            img = torch.from_numpy(img).contiguous()
            return img.to(dtype).div(255.0)

        if isinstance(img, Sequence):
            img = [_to_tensor(p) for p in img]
            img = torch.stack(img)
        else:
            img = _to_tensor(img)

        return img

    def get_frames_name_list(self, video_dict):
        video_list = []
        for vname, vlist in video_dict.items():
            samples_idx = list(range(0, len(vlist), self.stride))
            vlist = [vlist[i] for i in samples_idx]

            if len(vlist) < self.num_frames:
                vlist = vlist + [vlist[-1],] * (self.num_frames - len(vlist))
                video_list.append(vlist)
            else:
                num_samples = int(
                    np.ceil((len(vlist) - self.num_frames) / self.sample_stride + 1))

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

    def load_img_points_matting(self, item):
        img = join(
            self.train_data_dir, item["image"]) if self.train_data_dir is not None else item["image"]
        points = join(
            self.condition_data_dir, item["pose"]) if self.condition_data_dir is not None else item["pose"]
        matting = None
        if self.use_matting:
            matting = join(
                self.matting_data_dir, item["matting"]) if self.matting_data_dir is not None else item["matting"]

        if self.use_vos:
            img = self.vos.download_vos_pil(img)
            points = self.vos.download_vos_pkl(points)
            if self.use_matting:
                matting = self.vos.download_vos_pil(matting)
        else:
            points = pkl_load(points)

        img = load_image(img)
        if self.use_matting:
            matting = load_image(matting)

        return img, points, matting

    def load_ref_img_matting(self, item):
        ref_img = join(self.train_data_dir, item["image"]) if self.train_data_dir is not None else item["image"]
        ref_matting = None
        if self.use_matting:
            ref_matting = join(
                self.matting_data_dir, item["matting"]) if self.matting_data_dir is not None else item["matting"]

        if self.use_vos:
            ref_img = self.vos.download_vos_pil(ref_img)
            if self.use_matting:
                ref_matting = self.vos.download_vos_pil(ref_matting)

        ref_img = load_image(ref_img)
        if self.use_matting:
            ref_matting = load_image(ref_matting)

        return ref_img, ref_matting

    def get_frames(self, video_list, random_sample=False):
        images, condition_points = [], []
        mattings = []
        if random_sample:
            st_idx = random.randint(0, self.stride - 1)
            samples_idx = list(range(st_idx, len(video_list), self.stride))
            if len(samples_idx) >= self.num_frames:
                st_idx = random.randint(0, len(samples_idx) - self.num_frames)
                samples_idx = samples_idx[st_idx: st_idx + self.num_frames]
            else:
                samples_idx = samples_idx + [samples_idx[-1], ] * (self.num_frames - len(samples_idx))

            for i in samples_idx:
                item = video_list[i]
                img, points, matting = self.load_img_points_matting(item)
                images.append(img)
                condition_points.append(points)
                mattings.append(matting)
            ref_item = video_list[random.choice(samples_idx)]
        else:
            for item in video_list:
                img, points, matting = self.load_img_points_matting(item)
                images.append(img)
                condition_points.append(points)
                mattings.append(matting)
            ref_item = random.choice(video_list)

        reference_image, ref_matting = self.load_ref_img_matting(ref_item)

        return images, condition_points, reference_image, mattings, ref_matting

    def get_data(self, index):
        if self.is_video:
            # train with video
            vlist = copy.deepcopy(self.video_list[index])
            images, condition_points, reference_image, mattings, ref_matting = self.get_frames(vlist)
            data = {
                'image': images,
                'condition': condition_points,
                'reference_image': reference_image,
                'matting': mattings,
                'reference_matting': ref_matting,
            }
        else:
            # train with image
            video_name, item = self.item_list[index]
            img, points, matting = self.load_img_points_matting(item)
            # random select reference image
            video_list = copy.deepcopy(self.metadata[video_name])
            if len(video_list) > 2 * self.clip_length:
                idx = video_list.index(item)
                st_idx = idx - self.clip_length if idx - self.clip_length >= 0 else 0
                end_idx = idx + self.clip_length if idx + self.clip_length < len(video_list) else len(video_list)

                st_idx = len(video_list) - 2 * self.clip_length if end_idx == len(video_list) else st_idx
                end_idx = 2 * self.clip_length if st_idx == 0 else end_idx
                video_list = video_list[st_idx: end_idx]

            if len(video_list) > 1:
                video_list.remove(item)
            ref_item = random.choice(video_list)
            ref_img, ref_matting = self.load_ref_img_matting(ref_item)
            data = {
                'image': img,
                'condition': points,
                'reference_image': ref_img,
                'matting': matting,
                'reference_matting': ref_matting,
            }
        data = self.image_transforms(data)
        data['image'] = self.to_tensor(data['image'])
        data['condition_image'] = self.to_tensor(data['condition_image'])

        return data

    def __getitem__(self, index):
        example = {}
        data = self.get_data(index)

        example["pixel_values"] = self.img_normalize(data['image'])
        example["conditioning_pixel_values"] = data['condition_image']

        if random.random() < self.drop_text:
            text_inputs = self.empty_text_inputs
            data['reference_image'] = Image.new("RGB", data['reference_image'].size)
        else:
            text_inputs = self.tokenizer(
                self.caption, max_length=self.tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt"
            )

        example["reference_pixel_values"] = self.img_normalize(ToTensor()(data['reference_image']))
        example["input_ids"] = text_inputs.input_ids
        if self.clip_processor is not None:
            example["reference_image"] = self.clip_processor(
                images=data["reference_image"], return_tensors="pt").pixel_values

        return example


class PoseTransDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            dataset_file,
            train_data_dir,
            condition_data_dir,
            clip_processor=None,
            img_size=512,
            prob_uncond=0.1,
            matting_data_dir=None,
            use_matting=False,
            use_vos=False,
            clip_length=25,
    ):
        if prob_uncond < 0. or prob_uncond > 1.:
            raise ValueError("`prob_uncond` must be in the range [0., 1.].")

        self.train_data_dir = train_data_dir
        self.condition_data_dir = condition_data_dir
        self.clip_processor = clip_processor
        self.prob_uncond = prob_uncond
        self.matting_data_dir = matting_data_dir
        self.use_matting = use_matting
        self.use_vos = use_vos
        self.clip_length = clip_length

        self.metadata = self.get_metadata(dataset_file)

        self.item_list = []
        for k, items in tqdm(self.metadata.items()):
            self.item_list.extend([(k, item) for item in items])
        self._length = len(self.item_list)

        if use_vos:
            self.vos = VOSClient()

        self.image_transforms = tv_transforms.Compose(
            [
                DrawPose(prob_face=0.0),
                PasteMatting() if use_matting else tv_transforms.Lambda(lambda x: x),
                BoxCrop(img_size) if not isinstance(img_size, int) else tv_transforms.Lambda(lambda x: x),
                ResizePad(img_size, padding=isinstance(img_size, int)),
                RandomHorizontalFlip(),
            ]
        )
        self.img_normalize = Normalize([0.5], [0.5], inplace=True)
        self.random_hflip = RandomHorizontalFlip()

    def __len__(self):
        return self._length

    def get_metadata(self, dataset_file):
        print("Loading dataset...")
        if self.use_vos:
            dataset_file = [d.strip() for d in dataset_file.split(',') if len(d.strip()) > 0]
            out = {}
            for file_path in dataset_file:
                temp = load_file(file_path)
                out.update(temp)
        else:
            out = load_file(dataset_file)

        return out

    def to_tensor(self, img, dtype=torch.float32):
        def _to_tensor(img):
            img = np.array(img).transpose((2, 0, 1))
            img = torch.from_numpy(img).contiguous()
            return img.to(dtype).div(255.0)

        if isinstance(img, Sequence):
            img = [_to_tensor(p) for p in img]
            img = torch.stack(img)
        else:
            img = _to_tensor(img)

        return img

    def load_img_points_matting(self, item):
        img = join(
            self.train_data_dir, item["image"]) if self.train_data_dir is not None else item["image"]
        points = join(
            self.condition_data_dir, item["pose"]) if self.condition_data_dir is not None else item["pose"]
        matting = None
        if self.use_matting:
            matting = join(
                self.matting_data_dir, item["matting"]) if self.matting_data_dir is not None else item["matting"]

        if self.use_vos:
            img = self.vos.download_vos_pil(img)
            points = self.vos.download_vos_pkl(points)
            if self.use_matting:
                matting = self.vos.download_vos_pil(matting)
        else:
            points = pkl_load(points)

        img = load_image(img)
        if self.use_matting:
            matting = load_image(matting)

        return img, points, matting

    def load_ref_img_matting(self, item):
        ref_img = join(self.train_data_dir, item["image"]) if self.train_data_dir is not None else item["image"]
        ref_matting = None
        if self.use_matting:
            ref_matting = join(
                self.matting_data_dir, item["matting"]) if self.matting_data_dir is not None else item["matting"]

        if self.use_vos:
            ref_img = self.vos.download_vos_pil(ref_img)
            if self.use_matting:
                ref_matting = self.vos.download_vos_pil(ref_matting)

        ref_img = load_image(ref_img)
        if self.use_matting:
            ref_matting = load_image(ref_matting)

        return ref_img, ref_matting

    def get_data(self, index):
        # train with image
        video_name, item = self.item_list[index]
        img, points, matting = self.load_img_points_matting(item)
        # random select reference image
        video_list = copy.deepcopy(self.metadata[video_name])
        if len(video_list) > 2 * self.clip_length:
            idx = video_list.index(item)
            st_idx = idx - self.clip_length if idx - self.clip_length >= 0 else 0
            end_idx = idx + self.clip_length if idx + self.clip_length < len(video_list) else len(video_list)

            st_idx = len(video_list) - 2 * self.clip_length if end_idx == len(video_list) else st_idx
            end_idx = 2 * self.clip_length if st_idx == 0 else end_idx
            video_list = video_list[st_idx: end_idx]

        if len(video_list) > 1:
            video_list.remove(item)
        ref_item = random.choice(video_list)
        ref_img, ref_points, ref_matting = self.load_img_points_matting(ref_item)
        data = {
            'image': img,
            'condition': points,
            'matting': matting,
            'reference_image': ref_img,
            'reference_condition': ref_points,
            'reference_matting': ref_matting,
        }
        data = self.image_transforms(data)
        data['image'] = self.to_tensor(data['image'])
        data['condition_image'] = self.to_tensor(data['condition_image'])

        return data

    def __getitem__(self, index):
        example = {}
        data = self.get_data(index)

        example["pixel_values"] = self.img_normalize(data["image"])
        example["conditioning_pixel_values"] = data["condition_image"]
        reference_image = self.random_hflip(data["reference_image"])
        example["reference_pixel_values"] = self.img_normalize(self.to_tensor(reference_image))

        example["uncond"] = torch.tensor(
            [[[0.]]]) if random.random() < self.prob_uncond else torch.tensor([[[1.]]])

        if self.clip_processor is not None:
            example["reference_image"] = self.clip_processor(
                images=reference_image, return_tensors="pt").pixel_values

        return example


class ConXSPoseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_file,
        train_data_dir,
        condition_data_dir,
        clip_processor=None,
        img_size=512,
        prob_uncond=0.1,
        use_vos=False,
    ):
        if prob_uncond < 0. or prob_uncond > 1.:
            raise ValueError("`prob_uncond` must be in the range [0., 1.].")

        self.clip_processor = clip_processor
        self.prob_uncond = prob_uncond
        self.use_vos = use_vos

        self.metadata = self.get_metadata(dataset_file, train_data_dir, condition_data_dir)
        self._length = len(self.metadata)

        if use_vos:
            self.vos = VOSClient()

        self.image_transforms = tv_transforms.Compose(
            [
                Resize(img_size, interpolation=tv_transforms.InterpolationMode.LANCZOS),
                DrawPose(),
                RandomSelects([
                    tv_transforms.Compose([
                        Resize(img_size, interpolation=tv_transforms.InterpolationMode.LANCZOS),
                        RandomCrop(img_size),
                    ]),
                    ResizePad(img_size, padding=isinstance(img_size, int)),
                ]),
            ]
        )
        self.to_tensor = ToTensor()
        self.img_normalize = Normalize([0.5], [0.5], inplace=True)

    def __len__(self):
        return self._length

    def get_metadata(self, dataset_file, train_data_dir, condition_data_dir):
        print("Loading dataset...")
        dataset_file = [d.strip() for d in dataset_file.split(',') if len(d.strip()) > 0]

        out = []
        if self.use_vos:
            for file_ in dataset_file:
                temp_list = load_file(file_)
                if isinstance(temp_list, list):
                    out.extend(temp_list)
                elif isinstance(temp_list, dict):
                    for k, v in temp_list.items():
                        out.extend(v)
                else:
                    raise Exception(f"Error dataset_file: ({type(temp_list)}){file_}")
        else:
            train_data_dir = [d.strip() for d in train_data_dir.split(',') if len(d.strip()) > 0]
            condition_data_dir = [d.strip() for d in condition_data_dir.split(',') if len(d.strip()) > 0]
            for file_, tdir, cdir in zip(dataset_file, train_data_dir, condition_data_dir):
                temp_list = load_file(file_)
                for name, caption in tqdm(temp_list):
                    out.append({
                        "image": join(tdir, name),
                        "pose": join(cdir, splitext(name)[0] + '.pose'),
                        "caption": caption
                    })
        return out

    def load_img_points(self, item):
        if self.use_vos:
            img = self.vos.download_vos_pil(item["image"])
            img = load_image(img)
            points = self.vos.download_vos_pkl(item["pose"])
        else:
            img = load_image(item["image"])
            points = pkl_load(item["pose"])

        return img, points

    def __getitem__(self, index):
        example = {}
        item = self.metadata[index]

        img, points = self.load_img_points(item)
        data = {
            'image': img,
            'condition': points,
        }
        data = self.image_transforms(data)
        example["pixel_values"] = self.img_normalize(self.to_tensor(data["image"]))
        example["conditioning_pixel_values"] = self.to_tensor(data["condition_image"])

        if self.clip_processor is not None:
            example["reference_image"] = self.clip_processor(
                images=data["image"], return_tensors="pt").pixel_values

            example["uncond"] = torch.tensor(
                [[0.]]) if random.random() < self.prob_uncond else torch.tensor([[1.]])

        return example


class IPAPoseTransDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            dataset_file,
            train_data_dir,
            condition_data_dir=None,
            clip_processor=None,
            tokenizer=None,
            img_size=512,
            prob_uncond=0.1,
            use_vos=False,
    ):
        if prob_uncond < 0. or prob_uncond > 1.:
            raise ValueError("`prob_uncond` must be in the range [0., 1.].")

        self.clip_processor = clip_processor
        self.tokenizer = tokenizer
        self.prob_uncond = prob_uncond
        self.use_vos = use_vos

        if tokenizer is not None:
            self.empty_text_input_ids = tokenizer(
                "", max_length=tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt").input_ids

        self.metadata = self.get_metadata(dataset_file, train_data_dir, condition_data_dir)
        self.item_list = []
        for k, items in tqdm(self.metadata.items()):
            self.item_list.extend([(k, item) for item in items])
        self._length = len(self.item_list)

        if use_vos:
            self.vos = VOSClient()

        self.image_transforms = tv_transforms.Compose(
            [
                DrawPose(img_size),
                RandomSelects([
                    tv_transforms.Compose([
                        RandomResize(
                            img_size,
                            [img_size, int(img_size * 1.625)],
                            prob=0.5,
                            interpolation=tv_transforms.InterpolationMode.LANCZOS),
                        RandomCrop(img_size, fill=255, pad_if_needed=True),
                    ]),
                    ResizePad(img_size),
                ]),
                ToTensor(),
            ]
        )
        self.reference_transforms = tv_transforms.Compose(
            [
                HumanCrop(img_size, prob=1 - prob_uncond, random_crop=True),
                RandomHorizontalFlip(),
            ]
        )
        self.normalize_ = Normalize([0.5], [0.5], inplace=True)

    def __len__(self):
        return self._length

    def get_metadata(self, dataset_file, train_data_dir, condition_data_dir):
        print("Loading dataset...")
        dataset_file = [a.strip() for a in dataset_file.split(',') if len(a.strip()) > 0]

        out_dict = {}
        if self.use_vos:
            for file_ in dataset_file:
                temp_list = load_file(file_)
                if isinstance(temp_list, list):
                    for item in temp_list:
                        out_dict[get_str_md5(item["image"])] = [item]
                elif isinstance(temp_list, dict):
                    out_dict.update(temp_list)
                else:
                    raise Exception(f"Error dataset_file: ({type(temp_list)}){file_}")
        else:
            train_data_dir = [a.strip() for a in train_data_dir.split(',') if len(a.strip()) > 0]
            if condition_data_dir is not None:
                condition_data_dir = [
                    a.strip() for a in condition_data_dir.split(',') if len(a.strip()) > 0]

            for i, file_ in enumerate(dataset_file):
                temp_list = load_file(file_)
                for name, caption in tqdm(temp_list):
                    img_path = join(train_data_dir[i], name)
                    pose_path = join(
                            condition_data_dir[i], splitext(name)[0] + '.pose') if condition_data_dir is not None else None
                    out_dict[get_str_md5(img_path)] = [{
                        "image": img_path,
                        "caption": caption,
                        "pose": pose_path,
                    }]

        return out_dict

    def read_image_pose_data(self, item):
        img, pose = item["image"], item["pose"]
        if self.use_vos:
            img = self.vos.download_vos_pil(img)
            pose = self.vos.download_vos_pkl(pose)
        else:
            pose = pkl_load(item["pose"])

        img = load_image(img)
        return img, pose

    def get_text_ids(self, captions):
        text_ids = None
        if self.tokenizer is not None:
            if isinstance(captions, str):
                caption = captions
            elif isinstance(captions, Sequence):
                caption = random.choice(captions)
            else:
                raise Exception(f"Input captions type({type(captions)}) error.")

            text_ids = self.tokenizer(
                caption, max_length=self.tokenizer.model_max_length,
                padding="max_length", truncation=False, return_tensors="pt"
            ).input_ids
            if text_ids.shape[1] > self.tokenizer.model_max_length:
                len_text = self.tokenizer.model_max_length - 2
                st_idx = random.randint(1, text_ids.shape[1] - 1 - len_text)
                new_text_ids = text_ids[:, st_idx: st_idx + len_text]
                text_ids = torch.cat([text_ids[:, 0:1], new_text_ids, text_ids[:, -1:]], dim=1)

        return text_ids

    def get_data(self, index):
        group_name, item = self.item_list[index]
        img, pose = self.read_image_pose_data(item)
        data = {
            "image": img,
            "condition": pose,
        }

        # get image caption
        text_ids = None
        # captions = item.get("caption", "")
        # text_ids = self.get_text_ids(captions)

        # get class label
        class_label = item.get("goods_bbox", [])
        class_label = 1 if len(class_label) == 0 else 2

        if len(self.metadata[group_name]) > 1:
            group_list = copy.deepcopy(self.metadata[group_name])
            group_list.remove(item)
            ref_item = random.choice(group_list)

            ref_img, ref_pose = self.read_image_pose_data(ref_item)
            reference_data = {
                "reference_image": ref_img,
                "reference_condition": ref_pose,
            }
        else:
            reference_data = {
                "reference_image": img.copy(),
                "reference_condition": copy.deepcopy(pose),
            }

        return data, reference_data, text_ids, class_label

    def __getitem__(self, index):
        example = {}
        try:
            data, reference_data, text_input_ids, class_label = self.get_data(index)
        except:
            group_name, item = self.item_list[index]
            image_name = item["image"]
            print(f"error read: {image_name}.")
            index = random.randint(0, self._length - 1)
            data, reference_data, text_input_ids, class_label = self.get_data(index)
        data = self.image_transforms(data)
        example["pixel_values"] = self.normalize_(data["image"])
        example["conditioning_pixel_values"] = data["condition_image"]

        example["uncond"] = torch.tensor(
            [[0.]]) if random.random() < self.prob_uncond else torch.tensor([[1.]])
        # class_label = torch.LongTensor([class_label])
        # example["class_label"] = class_label

        if self.clip_processor is not None:
            reference_data = self.reference_transforms(reference_data)
            example["reference_image"] = self.clip_processor(
                images=reference_data["reference_image"], return_tensors="pt").pixel_values

        if self.tokenizer is not None:
            # if random.random() < self.prob_uncond:
            #     text_input_ids = self.empty_text_input_ids
            example["input_ids"] = self.empty_text_input_ids

        return example


class KolorsIPAPoseTransDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            dataset_file,
            clip_processor,
            img_size=512,
            prob_uncond=0.1,
    ):
        if prob_uncond < 0. or prob_uncond > 1.:
            raise ValueError("`prob_uncond` must be in the range [0., 1.].")

        self.clip_processor = clip_processor
        self.img_size = img_size
        self.prob_uncond = prob_uncond
        self.vos = VOSClient()

        self.metadata = self.get_metadata(dataset_file)
        self.item_list = []
        for key, items in tqdm(self.metadata.items()):
            self.item_list.extend([(key, item) for item in items])
        self._length = len(self.item_list)

        self.image_transforms = tv_transforms.Compose(
            [
                DrawPose(img_size),
                RandomSelects([
                    tv_transforms.Compose([
                        RandomResize(
                            img_size,
                            [img_size, int(img_size * 1.625)],
                            prob=0.5,
                            interpolation=tv_transforms.InterpolationMode.LANCZOS),
                        RandomCrop(img_size, fill=255, pad_if_needed=True),
                    ]),
                    ResizePad(img_size),
                ]),
                ToTensor(),
            ]
        )
        self.reference_transforms = tv_transforms.Compose(
            [
                HumanCrop(img_size, prob=0.8, random_crop=True),
                RandomHorizontalFlip(),
            ]
        )
        self.normalize_ = Normalize([0.5], [0.5], inplace=True)

    def __len__(self):
        return self._length

    def get_metadata(self, dataset_file):
        print("Loading dataset...")
        dataset_file = [a.strip() for a in dataset_file.split(',') if len(a.strip()) > 0]

        out_dict = {}
        for file_ in dataset_file:
            temp_list = load_file(file_)
            if isinstance(temp_list, list):
                for item in temp_list:
                    out_dict[get_str_md5(item["image"])] = [item]
            elif isinstance(temp_list, dict):
                out_dict.update(temp_list)
            else:
                raise Exception(f"Error dataset_file: ({type(temp_list)}){file_}")

        return out_dict

    def get_add_time_ids(self, original_size, crops_coords_top_left, target_size):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids])
        return add_time_ids

    def read_image_pose_data(self, item):
        img, pose = item["image"], item["pose"]
        img = self.vos.download_vos_pil(img)
        pose = self.vos.download_vos_pkl(pose)

        img = load_image(img)
        return img, pose

    def get_data(self, index):
        group_name, item = self.item_list[index]
        img, pose = self.read_image_pose_data(item)
        data = {
            "image": img,
            "condition": pose,
        }

        if len(self.metadata[group_name]) > 1:
            group_list = copy.deepcopy(self.metadata[group_name])
            group_list.remove(item)
            ref_item = random.choice(group_list)

            ref_img, ref_pose = self.read_image_pose_data(ref_item)
            reference_data = {
                "reference_image": ref_img,
                "reference_condition": ref_pose,
            }
        else:
            reference_data = {
                "reference_image": img.copy(),
                "reference_condition": copy.deepcopy(pose),
            }

        return data, reference_data

    def __getitem__(self, index):
        example = {}
        try:
            data, reference_data = self.get_data(index)
        except:
            item = self.item_list[index][1]
            image_name = item["image"]
            print(f"error read: {image_name}.")
            index = random.randint(0, self._length - 1)
            data, reference_data = self.get_data(index)
        data = self.image_transforms(data)

        # prepare unet, controlnet input
        example["pixel_values"] = self.normalize_(data["image"])
        example["conditioning_pixel_values"] = data["condition_image"]

        # prepare ip adapter input
        reference_data = self.reference_transforms(reference_data)
        example["reference_image"] = self.clip_processor(
            images=reference_data["reference_image"], return_tensors="pt").pixel_values

        add_time_ids = self.get_add_time_ids((self.img_size, ) * 2, (0, 0), (self.img_size, ) * 2)
        example["add_time_ids"] = add_time_ids

        example["uncond"] = torch.tensor(
            [[0.]]) if random.random() < self.prob_uncond else torch.tensor([[1.]])

        return example


class ImageVariationDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            dataset_file,
            feature_extractor=None,
            img_size=512,
            prob_uncond=0.1,
            vos_bucket='public',
    ):
        if prob_uncond < 0. or prob_uncond > 1.:
            raise ValueError("`prob_uncond` must be in the range [0., 1.].")

        self.feature_extractor = feature_extractor
        self.img_size = img_size
        self.prob_uncond = prob_uncond
        self.vos = VOSClient(vos_bucket)

        self.metadata = self.get_metadata(dataset_file)
        self._length = len(self.metadata)

        self.image_transforms = tv_transforms.Compose(
            [
                RandomSelects([
                    tv_transforms.Compose([
                        RandomResize(
                            img_size,
                            [int(img_size * 0.75), int(img_size * 1.5)],
                            prob=0.5,
                            interpolation=tv_transforms.InterpolationMode.LANCZOS),
                        RandomCrop(img_size, fill=255, pad_if_needed=True),
                    ]),
                    ResizePad(img_size),
                ]),
                RandomHorizontalFlip(),
            ]
        )
        self.normalize_ = tv_transforms.Compose([
            ToTensor(),
            Normalize([0.5], [0.5], inplace=True),
        ])

    def __len__(self):
        return self._length

    def get_metadata(self, dataset_file):
        print("Loading dataset...")
        dataset_file = [a.strip() for a in dataset_file.split(',') if len(a.strip()) > 0]

        out = []
        for file_ in dataset_file:
            temp_list = load_file(file_)
            if isinstance(temp_list, list):
                out += temp_list
            elif isinstance(temp_list, dict):
                for v in temp_list.values():
                    out += v
            else:
                raise Exception(f"Error dataset_file: ({type(temp_list)}){file_}")

        return out

    def get_data(self, index):
        item = self.metadata[index]
        img = self.vos.download_vos_pil(item["image"])
        img = load_image(img)
        return img

    def __getitem__(self, index):
        example = {}
        try:
            data = self.get_data(index)
        except:
            item = self.metadata[index]
            image_name = item["image"]
            print(f"error read: {image_name}.")
            index = random.randint(0, self._length - 1)
            data = self.get_data(index)
        data = self.image_transforms(data)

        # prepare unet, controlnet input
        example["pixel_values"] = self.normalize_(data)

        if self.feature_extractor is not None:
            # prepare ip adapter input
            example["reference_image"] = self.feature_extractor(
                images=data, return_tensors="pt").pixel_values

        example["uncond"] = torch.tensor(
            [[0.]]) if random.random() < self.prob_uncond else torch.tensor([[1.]])

        return example


class GroupImageVariationDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            dataset_file,
            feature_extractor=None,
            img_size=512,
            prob_uncond=0.1,
            vos_bucket='public',
    ):
        if prob_uncond < 0. or prob_uncond > 1.:
            raise ValueError("`prob_uncond` must be in the range [0., 1.].")

        self.feature_extractor = feature_extractor
        self.img_size = img_size
        self.prob_uncond = prob_uncond
        self.vos = VOSClient(vos_bucket)

        self.metadata = self.get_metadata(dataset_file)
        self.item_list = []
        for key, items in tqdm(self.metadata.items()):
            self.item_list.extend([(key, item) for item in items])
        self._length = len(self.item_list)

        self.image_transforms = tv_transforms.Compose([
            RandomSelects([
                tv_transforms.Compose([
                    RandomResize(
                        img_size,
                        [int(img_size * 0.75), int(img_size * 2.0)],
                        prob=1.0,
                        interpolation=tv_transforms.InterpolationMode.LANCZOS),
                    RandomCrop(img_size, fill=255, pad_if_needed=True),
                ]),
                ResizePad(img_size),
            ]),
            RandomHorizontalFlip(),
        ])

        self.normalize_ = tv_transforms.Compose([
            ToTensor(),
            Normalize([0.5], [0.5], inplace=True),
        ])

    def __len__(self):
        return self._length

    def get_metadata(self, dataset_file):
        print("Loading dataset...")
        dataset_file = [a.strip() for a in dataset_file.split(',') if len(a.strip()) > 0]

        out_dict = {}
        for file_ in dataset_file:
            temp_list = load_file(file_)
            if isinstance(temp_list, list):
                for item in temp_list:
                    out_dict[get_str_md5(item["image"])] = [item]
            elif isinstance(temp_list, dict):
                out_dict.update(temp_list)
            else:
                raise Exception(f"Error dataset_file: ({type(temp_list)}){file_}")

        return out_dict

    def read_image_data(self, item):
        img = self.vos.download_vos_pil(item["image"])
        return load_image(img)

    def get_data(self, index):
        group_name, item = self.item_list[index]
        img = self.read_image_data(item)

        if len(self.metadata[group_name]) > 1:
            group_list = copy.deepcopy(self.metadata[group_name])
            group_list.remove(item)
            ref_item = random.choice(group_list)

            ref_img = self.read_image_data(ref_item)
        else:
            ref_img = img.copy()

        return img, ref_img

    def __getitem__(self, index):
        example = {}
        data, reference_data = self.get_data(index)

        data = self.image_transforms(data)
        example["pixel_values"] = self.normalize_(data)

        # prepare ip adapter input
        reference_data = self.image_transforms(reference_data)
        example["reference_pixel_values"] = self.normalize_(reference_data)

        if self.feature_extractor is not None:
            example["reference_image"] = self.feature_extractor(
                images=reference_data, return_tensors="pt").pixel_values

        example["uncond"] = torch.tensor(
            [[0.]]) if random.random() < self.prob_uncond else torch.tensor([[1.]])

        return example


class FluxT2IDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            dataset_file,
            tokenizer_clip,
            tokenizer_t5,
            img_size=512,
            prob_uncond=0.1,
    ):
        if prob_uncond < 0. or prob_uncond > 1.:
            raise ValueError("`prob_uncond` must be in the range [0., 1.].")

        self.tokenizer_clip = tokenizer_clip
        self.tokenizer_t5 = tokenizer_t5
        self.img_size = img_size
        self.prob_uncond = prob_uncond
        self.vos = VOSClient()

        self.metadata = self.get_metadata(dataset_file)
        self._length = len(self.metadata)

        self.image_transforms = tv_transforms.Compose([
            RandomSelects([
                tv_transforms.Compose([
                    RandomResize(
                        img_size,
                        [int(img_size * 0.875), int(img_size * 1.625)],
                        prob=0.5,
                        interpolation=tv_transforms.InterpolationMode.LANCZOS),
                    RandomCrop(img_size, fill=255, pad_if_needed=True),
                ]),
                ResizePad(img_size),
            ]),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize([0.5], [0.5], inplace=True),
        ])

        self.empty_clip = tokenizer_clip(
            "",
            padding="max_length",
            max_length=77,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        ).input_ids
        self.empty_t5 = tokenizer_t5(
            "",
            padding="max_length",
            max_length=512,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        ).input_ids

    def __len__(self):
        return self._length

    def get_metadata(self, dataset_file):
        print("Loading dataset...")
        dataset_file = [a.strip() for a in dataset_file.split(',') if len(a.strip()) > 0]

        out = []
        for file_ in dataset_file:
            temp_list = load_file(file_)
            if isinstance(temp_list, list):
                out += temp_list
            elif isinstance(temp_list, dict):
                for v in temp_list.values():
                    out += v
            else:
                raise Exception(f"Error dataset_file: ({type(temp_list)}){file_}")

        return out

    def get_data(self, index):
        item = self.metadata[index]
        img = self.vos.download_vos_pil(item["image"])
        img = load_image(img)
        caption = item.get("caption", "")
        if isinstance(caption, (tuple, list)):
            caption = random.choice(caption)
        return img, caption

    def __getitem__(self, index):
        example = {}
        try:
            data, caption = self.get_data(index)
        except:
            item = self.metadata[index]
            image_name = item["image"]
            print(f"error read: {image_name}.")
            index = random.randint(0, self._length - 1)
            data, caption = self.get_data(index)
        data = self.image_transforms(data)

        example["pixel_values"] = data

        if random.random() < self.prob_uncond:
            example["uncond"] = torch.tensor([[0.]])
            example["input_ids"] = self.empty_t5
            example["pooled_input_ids"] = self.empty_clip
        else:
            example["uncond"] = torch.tensor([[1.]])
            example["input_ids"] = self.tokenizer_t5(
                caption,
                padding="max_length",
                max_length=512,
                truncation=True,
                return_length=False,
                return_overflowing_tokens=False,
                return_tensors="pt",
            ).input_ids
            example["pooled_input_ids"] = self.tokenizer_clip(
                caption,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_length=False,
                return_overflowing_tokens=False,
                return_tensors="pt",
            ).input_ids

        return example


class FluxPTDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            dataset_file,
            feature_extractor=None,
            img_size=512,
            prob_uncond=0.1,
            vos_bucket='public',
    ):
        if prob_uncond < 0. or prob_uncond > 1.:
            raise ValueError("`prob_uncond` must be in the range [0., 1.].")

        self.feature_extractor = feature_extractor
        self.img_size = img_size
        self.prob_uncond = prob_uncond
        self.vos = VOSClient(vos_bucket)

        self.metadata = self.get_metadata(dataset_file)
        self.item_list = []
        for key, items in tqdm(self.metadata.items()):
            self.item_list.extend([(key, item) for item in items])
        self._length = len(self.item_list)

        self.image_transforms = tv_transforms.Compose([
            DrawPose(1024, random_ratio=(0.75, 1.0)),
            RandomSelects([
                tv_transforms.Compose([
                    RandomResize(
                        img_size,
                        [int(img_size * 1.0), int(img_size * 2.0)],
                        prob=0.5,
                        interpolation=tv_transforms.InterpolationMode.LANCZOS),
                    RandomCrop(img_size, fill=255, pad_if_needed=True),
                ]),
                ResizePad(img_size),
            ]),
            ToTensor(),
            Normalize([0.5], [0.5], inplace=True),
        ])

        self.reference_transforms = tv_transforms.Compose([
            RandomSelects([
                tv_transforms.Compose([
                    RandomResize(
                        img_size,
                        [int(img_size * 0.75), int(img_size * 1.25)],
                        prob=0.5,
                        interpolation=tv_transforms.InterpolationMode.LANCZOS),
                    RandomCrop(img_size, fill=255, pad_if_needed=True),
                ]),
                HumanCrop(img_size, prob=0.5, random_crop=True),
            ]),
        ])

        self.normalize_ = tv_transforms.Compose([
            ToTensor(),
            Normalize([0.5], [0.5], inplace=True),
        ])

    def __len__(self):
        return self._length

    def get_metadata(self, dataset_file):
        print("Loading dataset...")
        dataset_file = [a.strip() for a in dataset_file.split(',') if len(a.strip()) > 0]

        out_dict = {}
        for file_ in dataset_file:
            temp_list = load_file(file_)
            if isinstance(temp_list, list):
                for item in temp_list:
                    out_dict[get_str_md5(item["image"])] = [item]
            elif isinstance(temp_list, dict):
                out_dict.update(temp_list)
            else:
                raise Exception(f"Error dataset_file: ({type(temp_list)}){file_}")

        return out_dict

    def read_image_pose_data(self, item):
        img, pose = item["image"], item["pose"]
        img = self.vos.download_vos_pil(img)
        pose = self.vos.download_vos_pkl(pose)

        img = load_image(img)
        return img, pose

    def get_data(self, index):
        group_name, item = self.item_list[index]
        img, pose = self.read_image_pose_data(item)
        data = {
            "image": img,
            "condition": pose,
        }

        if len(self.metadata[group_name]) > 1:
            group_list = copy.deepcopy(self.metadata[group_name])
            group_list.remove(item)
            ref_item = random.choice(group_list)

            ref_img, ref_pose = self.read_image_pose_data(ref_item)
            reference_data = {
                "image": ref_img,
                "condition": ref_pose,
            }
        else:
            reference_data = {
                "image": img.copy(),
                "condition": copy.deepcopy(pose),
            }

        return data, reference_data

    def __getitem__(self, index):
        example = {}
        try:
            data, reference_data = self.get_data(index)
        except:
            item = self.metadata[index]
            image_name = item["image"]
            print(f"error read: {image_name}.")
            index = random.randint(0, self._length - 1)
            data, reference_data = self.get_data(index)
        data = self.image_transforms(data)

        # prepare unet, controlnet input
        example["pixel_values"] = data["image"]
        example["conditioning_pixel_values"] = data["condition_image"]

        # prepare ip adapter input
        reference_data = self.reference_transforms(reference_data)
        example["reference_pixel_values"] = self.normalize_(reference_data["image"])

        if self.feature_extractor is not None:
            example["reference_image"] = self.feature_extractor(
                images=reference_data["image"], return_tensors="pt").pixel_values

        example["uncond"] = torch.tensor(
            [[0.]]) if random.random() < self.prob_uncond else torch.tensor([[1.]])

        return example


class FLUXICImageDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            dataset_file,
            prob_uncond=0.1,
            vos_bucket='public',
    ):
        if prob_uncond < 0. or prob_uncond > 1.:
            raise ValueError("`prob_uncond` must be in the range [0., 1.].")

        self.prob_uncond = prob_uncond
        self.vos = VOSClient(vos_bucket)

        self.metadata = self.get_metadata(dataset_file)
        self._length = len(self.metadata)

        root_dir, _ = split(self.metadata[0]["latents"])
        s3_path = root_dir + '/pooled_prompt_embeds.text'
        self.pooled_prompt_embeds = self.vos.download_vos_pt(s3_path)

    def __len__(self):
        return self._length

    def get_metadata(self, dataset_file):
        print("Loading dataset...")
        dataset_file = [a.strip() for a in dataset_file.split(',') if len(a.strip()) > 0]

        out = []
        for file_ in dataset_file:
            temp_list = load_file(file_)
            if isinstance(temp_list, list):
                out += temp_list
            elif isinstance(temp_list, dict):
                for v in temp_list.values():
                    out += v
            else:
                raise Exception(f"Error dataset_file: ({type(temp_list)}){file_}")

        return out

    def get_data(self, index):
        item = self.metadata[index]
        latents = self.vos.download_vos_pt(item['latents'])[0]
        prompt_embeds = self.vos.download_vos_pt(item['prompt_embeds'])
        if prompt_embeds.shape[0] > 1:
            idx = random.choice(range(prompt_embeds.shape[0]))
            prompt_embeds = prompt_embeds[idx: idx + 1]

        return latents, prompt_embeds

    def __getitem__(self, index):
        example = {}
        latents, prompt_embeds = self.get_data(index)

        example["pixel_values"] = latents
        example["prompt_embeds"] = prompt_embeds
        example["pooled_prompt_embeds"] = self.pooled_prompt_embeds.detach()

        example["uncond"] = torch.tensor(
            [[0.]]) if random.random() < self.prob_uncond else torch.tensor([[1.]])

        return example


class FluxFillDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            dataset_file,
            feature_extractor=None,
            batch_size=1,
            img_size=1024,
            vae_scale_factor=16,
            vos_bucket='public',
    ):
        assert batch_size == 1, "`batch_size` must be 1."

        self.feature_extractor = feature_extractor
        self.img_size = img_size
        self.vae_scale_factor = vae_scale_factor
        self.vos = VOSClient(vos_bucket)

        self.metadata = self.get_metadata(dataset_file)
        self._length = len(self.metadata)

        self.image_transforms = tv_transforms.Compose([
            RandomSelects([
                RandomRectMask(size=img_size, scale_factor=vae_scale_factor),
                RandomHumanMask(size=img_size, scale_factor=vae_scale_factor),
            ]),
            ToTensor(),
            Normalize([0.5], [0.5], inplace=True),
        ])

    def __len__(self):
        return self._length

    def get_metadata(self, dataset_file):
        print("Loading dataset...")
        dataset_file = [a.strip() for a in dataset_file.split(',') if len(a.strip()) > 0]

        out_list = []
        for file_ in dataset_file:
            temp_list = load_file(file_)
            if isinstance(temp_list, list):
                out_list += temp_list
            elif isinstance(temp_list, dict):
                for v in temp_list.values():
                    out_list += v
            else:
                raise Exception(f"Error dataset_file: ({type(temp_list)}){file_}")

        return out_list

    def get_data(self, index):
        item = self.metadata[index]
        img = self.vos.download_vos_pil(item["image"])
        data = {"image": img}

        if "pose" in item:
            data["pose"] = self.vos.download_vos_pkl(item["pose"])
        # if "matting" in item:
        #     data["matting"] = self.vos.download_vos_pil(item["matting"])
        return data

    def _prepare_latent_image_ids(self, height, width):
        height = height // self.vae_scale_factor
        width = width // self.vae_scale_factor
        latent_image_ids = torch.zeros(height, width, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )

        return latent_image_ids

    @staticmethod
    def _prepare_mask(mask):
        if isinstance(mask, Image.Image):
            mask = torch.from_numpy(np.array(mask))
        elif isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)

        h, w = mask.shape[:2]
        h = h // 8
        w = w // 8

        mask = mask.view(h, 8, w, 8)  # height, 8, width, 8
        mask = mask.permute(1, 3, 0, 2)  # 8, 8, height, width
        mask = mask.reshape(8 * 8, h, w)  # 8*8, height, width

        return mask

    def __getitem__(self, index):
        example = {}
        data = self.get_data(index)
        data = self.image_transforms(data)

        example["pixel_values"] = data["image"]
        example["masked_image_pixel_values"] = data["image"] * (1 - data["mask"])
        example["mask"] = self._prepare_mask(data["mask"])
        example["latent_image_ids"] = self._prepare_latent_image_ids(*data["image"].shape[1:])

        return example


class FluxAnyPTDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            dataset_file,
            feature_extractor=None,
            batch_size=1,
            img_size=1200,
            prob_uncond=0.1,
            vae_scale_factor=16,
            max_sequence_length=512,
            pad_square=False,
            condition='pose',
            use_text=False,
            vos_bucket='public',
    ):
        assert batch_size == 1, "`batch_size` must be 1."
        assert condition in ['pose', 'depth']

        if prob_uncond < 0. or prob_uncond > 1.:
            raise ValueError("`prob_uncond` must be in the range [0., 1.].")

        self.feature_extractor = feature_extractor
        self.img_size = img_size
        self.prob_uncond = prob_uncond
        self.vae_scale_factor = vae_scale_factor
        self.max_sequence_length = max_sequence_length
        self.condition = condition
        self.use_text = use_text
        self.vos = VOSClient(vos_bucket)

        self.metadata = self.get_metadata(dataset_file)
        self.item_list = []
        for key, items in tqdm(self.metadata.items()):
            self.item_list.extend([(key, item) for item in items])
        self._length = len(self.item_list)

        self.image_transforms = tv_transforms.Compose([
            DrawPose(1024, random_ratio=(0.75, 1.0)) if condition == 'pose' else tv_transforms.Lambda(lambda x: x),
            PTRandomCrop(
                img_size,
                prob=0.2,
                condition=condition,
                pad_square=pad_square,
                scale_factor=vae_scale_factor
            ),
        ])

        self.normalize_ = tv_transforms.Compose([
            ToTensor(),
            Normalize([0.5], [0.5], inplace=True),
        ])

    def __len__(self):
        return self._length

    def get_metadata(self, dataset_file):
        print("Loading dataset...")
        dataset_file = [a.strip() for a in dataset_file.split(',') if len(a.strip()) > 0]

        out_dict = {}
        for file_ in dataset_file:
            temp_list = load_file(file_)
            if isinstance(temp_list, list):
                for item in temp_list:
                    out_dict[get_str_md5(item["image"])] = [item]
            elif isinstance(temp_list, dict):
                out_dict.update(temp_list)
            else:
                raise Exception(f"Error dataset_file: ({type(temp_list)}){file_}")

        return out_dict

    def read_image_condition_data(self, item, use_text=False):
        img = self.vos.download_vos_pil(item["image"])
        img = load_image(img)

        if self.condition == 'pose':
            condition = self.vos.download_vos_pkl(item["pose"])
        else:
            condition = self.vos.download_vos_pil(item["depth"])
            condition = load_image(condition)

        text = None
        if use_text and self.use_text:
            text = self.vos.download_vos_pt(item["text_embeds"])

        return img, condition, text

    def get_data(self, index):
        group_name, item = self.item_list[index]
        img, condition, text = self.read_image_condition_data(item, use_text=True)
        data = {
            "image": img,
            "condition": condition,
            "text": text,
        }

        if len(self.metadata[group_name]) > 1:
            group_list = copy.deepcopy(self.metadata[group_name])
            group_list.remove(item)
            ref_item = random.choice(group_list)

            ref_img, ref_condition, _ = self.read_image_condition_data(ref_item)
            reference_data = {
                "image": ref_img,
                "condition": ref_condition,
            }
        else:
            reference_data = {
                "image": img.copy(),
                "condition": copy.deepcopy(condition) if self.condition == 'pose' else condition.copy(),
            }

        return data, reference_data

    def _prepare_latent_image_ids(self, height, width, idx=0):
        height = height // self.vae_scale_factor
        width = width // self.vae_scale_factor
        latent_image_ids = torch.zeros(height, width, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

        latent_image_ids = latent_image_ids.reshape(-1, 3)

        latent_image_ids[:, 0] = idx
        if self.use_text and idx == 1:
            txt_ids = torch.zeros(self.max_sequence_length, 3)
            latent_image_ids = torch.cat([txt_ids, latent_image_ids], dim=0)

        return latent_image_ids

    def __getitem__(self, index):
        example = {}
        data, reference_data = self.get_data(index)
        data = self.image_transforms(data)

        # prepare input
        example["pixel_values"] = self.normalize_(data["image"])
        example["conditioning_pixel_values"] = self.normalize_(data["condition_image"])
        if self.use_text:
            example["prompt_embeds"] = data["text"]["prompt_embeds"]
            example["pooled_prompt_embeds"] = data["text"]["pooled_prompt_embeds"]

        # prepare reference input
        reference_data = self.image_transforms(reference_data)
        example["reference_pixel_values"] = self.normalize_(reference_data["image"])

        if self.feature_extractor is not None:
            example["reference_image"] = self.feature_extractor(
                images=reference_data["image"], return_tensors="pt").pixel_values

        example["uncond"] = torch.tensor(
            [[0.]]) if random.random() < self.prob_uncond else torch.tensor([[1.]])

        w, h = data["image"].size
        example["img_ids"] = self._prepare_latent_image_ids(h, w, idx=0)
        w, h = reference_data["image"].size
        example["txt_ids"] = self._prepare_latent_image_ids(h, w, idx=1)

        return example


class FluxFillICT2IDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            dataset_file,
            tokenizer_clip,
            tokenizer_t5,
            batch_size=1,
            max_sequence_length=512,
            img_size=(1280, 960),
            vae_scale_factor=16,
            vos_bucket='public',
    ):
        self.tokenizer_clip = tokenizer_clip
        self.tokenizer_t5 = tokenizer_t5
        self.max_sequence_length = max_sequence_length
        self.img_size = img_size
        self.vae_scale_factor = vae_scale_factor
        self.vos = VOSClient(vos_bucket)

        self.metadata = self.get_metadata(dataset_file)
        self.item_list = []
        for key, items in tqdm(self.metadata.items()):
            self.item_list.extend([(key, item) for item in items])
        self._length = len(self.item_list)

        self.image_transforms = tv_transforms.Compose([
            FLUXICFillProc(img_size),
            ToTensor(),
            Normalize([0.5], [0.5], inplace=True),
        ])

    def __len__(self):
        return self._length

    def get_metadata(self, dataset_file):
        print("Loading dataset...")
        dataset_file = [a.strip() for a in dataset_file.split(',') if len(a.strip()) > 0]

        out_dict = {}
        for file_ in dataset_file:
            temp_list = load_file(file_)
            if isinstance(temp_list, list):
                for item in temp_list:
                    out_dict[get_str_md5(item["image"])] = [item]
            elif isinstance(temp_list, dict):
                out_dict.update(temp_list)
            else:
                raise Exception(f"Error dataset_file: ({type(temp_list)}){file_}")

        return out_dict

    def read_image_pose_data(self, item):
        img, pose = item["image"], item["pose"]
        img = self.vos.download_vos_pil(img)
        pose = self.vos.download_vos_pkl(pose)

        img = load_image(img)
        return img, pose

    def get_data(self, index):
        group_name, item = self.item_list[index]
        img, pose = self.read_image_pose_data(item)
        caption = item["caption"]
        data = {
            "image": img,
            "condition": pose,
        }

        if len(self.metadata[group_name]) > 1:
            group_list = copy.deepcopy(self.metadata[group_name])
            group_list.remove(item)
            ref_item = random.choice(group_list)

            ref_img, ref_pose = self.read_image_pose_data(ref_item)
            reference_data = {
                "image": ref_img,
                "condition": ref_pose,
            }
        else:
            reference_data = {
                "image": img.copy(),
                "condition": copy.deepcopy(pose),
            }

        return data, reference_data, caption

    def _prepare_latent_image_ids(self, height, width):
        height = height // self.vae_scale_factor
        width = width // self.vae_scale_factor
        latent_image_ids = torch.zeros(height, width * 2, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width * 2)[None, :]

        latent_image_ids[:, width:, 0] = 1

        return latent_image_ids.reshape(-1, 3)

    @staticmethod
    def _prepare_mask(h, w):
        mask = torch.cat([torch.ones([h, w]), torch.zeros([h, w])], dim=-1)

        h = h // 8
        w = w // 8
        mask = mask.view(h, 8, w * 2, 8)  # height, 8, width, 8
        mask = mask.permute(1, 3, 0, 2)  # 8, 8, height, width
        mask = mask.reshape(8 * 8, h, w * 2)  # 8*8, height, width

        return mask

    def __getitem__(self, index):
        example = {}
        data, reference_data, caption = self.get_data(index)
        data = self.image_transforms(data)
        reference_data = self.image_transforms(reference_data)

        example["pixel_values"] = data["image"]
        example["reference_pixel_values"] = reference_data["image"]
        example["mask"] = self._prepare_mask(*data["image"].shape[1:])
        example["latent_image_ids"] = self._prepare_latent_image_ids(*data["image"].shape[1:])

        example["input_ids"] = self.tokenizer_t5(
            caption,
            padding="max_length",
            max_length=self.max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        ).input_ids
        example["pooled_input_ids"] = self.tokenizer_clip(
            caption,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        ).input_ids

        return example


class FluxTextPTDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            dataset_file,
            feature_extractor=None,
            batch_size=1,
            img_size=1200,
            prob_uncond=0.1,
            vae_scale_factor=16,
            max_sequence_length=512,
            vos_bucket='public',
            crop_size=(1024, 768),
    ):
        assert batch_size == 1, "`batch_size` must be 1."

        if prob_uncond < 0. or prob_uncond > 1.:
            raise ValueError("`prob_uncond` must be in the range [0., 1.].")

        self.feature_extractor = feature_extractor
        self.img_size = img_size
        self.prob_uncond = prob_uncond
        self.vae_scale_factor = vae_scale_factor
        self.max_sequence_length = max_sequence_length
        self.vos = VOSClient(vos_bucket)

        self.metadata = self.get_metadata(dataset_file)
        self.item_list = []
        for key, items in tqdm(self.metadata.items()):
            self.item_list.extend([(key, item) for item in items])
        self._length = len(self.item_list)

        self.image_transforms = tv_transforms.Compose([
            TextPTRandomCrop(img_size, crop_size=crop_size, scale_factor=vae_scale_factor),
        ])

        self.normalize_ = tv_transforms.Compose([
            ToTensor(),
            Normalize([0.5], [0.5], inplace=True),
        ])

    def __len__(self):
        return self._length

    def get_metadata(self, dataset_file):
        print("Loading dataset...")
        dataset_file = [a.strip() for a in dataset_file.split(',') if len(a.strip()) > 0]

        out_dict = {}
        for file_ in dataset_file:
            temp_list = load_file(file_)
            if isinstance(temp_list, list):
                for item in temp_list:
                    out_dict[get_str_md5(item["image"])] = [item]
            elif isinstance(temp_list, dict):
                out_dict.update(temp_list)
            else:
                raise Exception(f"Error dataset_file: ({type(temp_list)}){file_}")

        return out_dict

    def _prepare_latent_image_ids(self, height, width, idx=0):
        height = height // self.vae_scale_factor
        width = width // self.vae_scale_factor
        latent_image_ids = torch.zeros(height, width, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

        latent_image_ids = latent_image_ids.reshape(-1, 3)

        latent_image_ids[:, 0] = idx
        if idx == 1:
            txt_ids = torch.zeros(self.max_sequence_length, 3)
            latent_image_ids = torch.cat([txt_ids, latent_image_ids], dim=0)

        return latent_image_ids

    def read_image_condition_data(self, item, use_text=False):
        img = self.vos.download_vos_pil(item["image"])
        img = load_image(img)

        condition = self.vos.download_vos_pkl(item["pose"])

        text = None
        if use_text:
            text = self.vos.download_vos_pt(item["text_embeds"])

        return img, condition, text

    def get_data(self, index):
        group_name, item = self.item_list[index]
        img, condition, text = self.read_image_condition_data(item, use_text=True)
        data = {
            "image": img,
            "condition": condition,
            "text": text,
        }

        if len(self.metadata[group_name]) > 1:
            group_list = copy.deepcopy(self.metadata[group_name])
            group_list.remove(item)
            ref_item = random.choice(group_list)

            ref_img, ref_condition, _ = self.read_image_condition_data(ref_item)
            reference_data = {
                "image": ref_img,
                "condition": ref_condition,
            }
        else:
            reference_data = {
                "image": img.copy(),
                "condition": copy.deepcopy(condition),
            }

        return data, reference_data

    def __getitem__(self, index):
        example = {}
        data, reference_data = self.get_data(index)
        data = self.image_transforms(data)

        # prepare input
        example["pixel_values"] = self.normalize_(data["image"])
        example["prompt_embeds"] = data["text"]["prompt_embeds"]
        example["pooled_prompt_embeds"] = data["text"]["pooled_prompt_embeds"]

        # prepare reference input
        reference_data['crop_image'] = True
        reference_data = self.image_transforms(reference_data)
        example["reference_pixel_values"] = self.normalize_(reference_data["image"])

        if self.feature_extractor is not None:
            example["reference_image"] = self.feature_extractor(
                images=reference_data["image"], return_tensors="pt").pixel_values

        example["uncond"] = torch.tensor(
            [[0.]]) if random.random() < self.prob_uncond else torch.tensor([[1.]])

        w, h = data["image"].size
        example["img_ids"] = self._prepare_latent_image_ids(h, w, idx=0)
        w, h = reference_data["image"].size
        example["txt_ids"] = self._prepare_latent_image_ids(h, w, idx=1)

        return example


class WanFLF2VDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            dataset_file,
            feature_extractor=None,
            img_size=(1280, 720),
            prob_uncond=0.1,
            max_sequence_length=512,
            vos_bucket='public',
            video_sample_stride=2,
            max_num_frames=49,
            vae_scale_factor_temporal=4,
            vae_scale_factor=16,
            use_latents=False,
            use_vos=False,
    ):
        if prob_uncond < 0. or prob_uncond > 1.:
            raise ValueError("`prob_uncond` must be in the range [0., 1.].")

        self.feature_extractor = feature_extractor
        self.img_size = img_size
        self.prob_uncond = prob_uncond
        self.max_sequence_length = max_sequence_length
        self.video_sample_stride = video_sample_stride
        self.max_num_frames = max_num_frames
        self.vae_scale_factor_temporal = vae_scale_factor_temporal
        self.vae_scale_factor = vae_scale_factor
        self.use_latents = use_latents
        self.vos = VOSClient(vos_bucket)
        self.use_vos = use_vos

        self.metadata = self.get_metadata(dataset_file)
        self._length = len(self.metadata)

    def __len__(self):
        return self._length

    def get_metadata(self, dataset_file):
        print("Loading dataset...")
        dataset_file = [a.strip() for a in dataset_file.split(',') if len(a.strip()) > 0]

        out_list = []
        for file_ in dataset_file:
            temp_list = load_file(file_)
            if isinstance(temp_list, list):
                out_list += temp_list
            else:
                raise Exception(f"Error dataset_file: ({type(temp_list)}){file_}")

        if self.use_latents:
            new_list = []
            for item in out_list:
                for line in item['latents']:
                    new_list.append({
                        "prompt_embeds": item["prompt_embeds"],
                        "latents": line
                    })
            out_list = new_list

        return out_list

    def load_prompt_embeds(self, text_encoder):
        device = text_encoder.device

        text_inputs = self.tokenizer(
            "",
            padding="max_length",
            max_length=self.max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()
        negative_prompt_embeds = text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
        negative_prompt_embeds = negative_prompt_embeds[:, :seq_lens]
        self.negative_prompt_embeds = torch.cat(
            [negative_prompt_embeds,
             negative_prompt_embeds.new_zeros(1, self.max_sequence_length - seq_lens, prompt_embeds.size(2))],
            dim=1
        )

        for item in tqdm(self.metadata):
            text_inputs = self.tokenizer(
                item['text'],
                padding="max_length",
                max_length=self.max_sequence_length,
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
            seq_lens = mask.gt(0).sum(dim=1).long()
            prompt_embeds = text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
            prompt_embeds = prompt_embeds[:, :seq_lens]
            prompt_embeds = torch.cat(
                [prompt_embeds, prompt_embeds.new_zeros(1, self.max_sequence_length - seq_lens, prompt_embeds.size(2))],
                dim=1
            )
            item['prompt_embeds'] = prompt_embeds.to(torch.device('cpu'))

    def aspect_ratio_resize(self, frames, mod_value=16):
        h, w = frames.shape[1:3]
        aspect_ratio = h / w
        max_area = self.img_size[0] * self.img_size[1]
        height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
        frames = frames.permute([0, 3, 1, 2]).to(torch.float32)
        frames = torch.nn.functional.interpolate(frames, size=[height, width], mode="bilinear")
        frames = frames / 255.0 * 2.0 - 1.0
        return frames

    def get_data(self, index):
        item = self.metadata[index]
        prompt_embeds = torch.load(item['prompt_embeds'], weights_only=True)

        video_reader = decord.VideoReader(item['file_path'])
        video_length = len(video_reader)

        if video_length < self.max_num_frames * self.video_sample_stride:
            clip_length = min(video_length, self.max_num_frames)
            min_sample_n_frames = clip_length
        else:
            min_sample_n_frames = min(self.max_num_frames, int(video_length // self.video_sample_stride))
            clip_length = min(video_length, (min_sample_n_frames - 1) * self.video_sample_stride + 1)

        start_idx = random.randint(0, video_length - clip_length) if video_length != clip_length else 0
        batch_index = np.linspace(start_idx, start_idx + clip_length - 1, min_sample_n_frames, dtype=int)

        sample_video_length = len(batch_index)
        sample_video_length = (sample_video_length - 1) // self.vae_scale_factor_temporal * \
                self.vae_scale_factor_temporal + 1
        batch_index = batch_index[:sample_video_length]
        video = video_reader.get_batch(batch_index)
        video = self.aspect_ratio_resize(video)
        images = [video[0], video[-1]]
        images = [a.permute([1, 2, 0]) for a in images]
        images = [Image.fromarray(np.uint8((a + 1) / 2 * 255)) for a in images]

        return video.permute([1, 0, 2, 3]), prompt_embeds, images

    def __getitem__(self, index):
        example = {}
        if not self.use_latents:
            data, prompt_embeds, images = self.get_data(index)

            # prepare input
            example["pixel_values"] = data
            example["reference_pixel_values"] = torch.cat(
                    [data[:, 0:1], data.new_zeros(data.shape[0], data.shape[1] - 2, data.shape[2], data.shape[3]), data[:, -1:]],
                    dim=1,
                )
            example["prompt_embeds"] = prompt_embeds

            example["reference_image"] = self.feature_extractor(
                images=images, return_tensors="pt").pixel_values
        else:
            item = self.metadata[index]

            if not self.use_vos:
                example["prompt_embeds"] = torch.load(item['prompt_embeds'], weights_only=True)
                latents = torch.load(item['latents'], weights_only=True)
            else:
                example["prompt_embeds"] = self.vos.download_vos_pt(item['prompt_embeds'])
                latents = self.vos.download_vos_pt(item['latents'])

            example["pixel_values"] = latents["latents"][0]
            example["reference_pixel_values"] = latents["condition"][0]
            example["reference_image"] = latents["image_embeds"]
            example["latents_mask"] = latents["mask"][0]

        example["uncond"] = torch.tensor(
            [[0.]]) if random.random() < self.prob_uncond else torch.tensor([[1.]])

        return example


class FluxKontextDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            dataset_file,
    ):
        self.metadata = self.get_metadata(dataset_file)
        self._length = len(self.metadata)

    def __len__(self):
        return self._length

    def get_metadata(self, dataset_file):
        print("Loading dataset...")
        dataset_file = [a.strip() for a in dataset_file.split(',') if len(a.strip()) > 0]

        data_list = {}
        for file_ in dataset_file:
            temp_list = load_file(file_)
            if isinstance(temp_list, dict):
                data_list.update(temp_list)
            else:
                raise Exception(f"Error dataset_file: ({type(temp_list)}){file_}")
        self.dataset_file = data_list

        out_list = [
            {"text_embedding": v["text_embedding"], "latents": v["latents"]}
            for v in data_list.values()
        ]

        return out_list

    def __getitem__(self, index):
        example = {}
        item = self.metadata[index]

        text_embedding = torch.load(item['text_embedding'], weights_only=True)
        example["prompt_embeds"] = text_embedding["prompt_embeds"]
        example["pooled_prompt_embeds"] = text_embedding["pooled_prompt_embeds"]

        latents = torch.load(item['latents'], weights_only=True)
        example["image_latents"] = latents["image_latents"][0]
        example["cond_image_latents"] = latents["cond_image_latents"][0]
        example["latent_image_ids"] = latents["latent_image_ids"]
        example["cond_image_ids"] = latents["cond_image_ids"]

        return example
