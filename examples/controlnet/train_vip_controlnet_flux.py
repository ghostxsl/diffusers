#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# Copyright 2024 The VIP Inc. AIGC team. All rights reserved.
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

import argparse
import logging
import math
import os
from os.path import join, exists
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, ProjectConfiguration, set_seed
from packaging import version
from tqdm.auto import tqdm
from transformers import (
    CLIPTokenizer,
    PretrainedConfig,
    T5TokenizerFast,
)

import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxTransformer2DModel,
    FluxPipeline,
)
from diffusers.models.controlnet_flux import FluxControlNetModel
from diffusers.optimization import get_scheduler
from diffusers.pipelines.flux.pipeline_flux_controlnet import FluxControlNetPipeline
from diffusers.training_utils import compute_density_for_timestep_sampling, free_memory
from diffusers.utils.import_utils import is_torch_npu_available, is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.data import ControlNetDataset, flux_collate_fn, ModelEMA
from diffusers.pipelines.flux.pipeline_flux import calculate_shift


logger = get_logger(__name__)
if is_torch_npu_available():
    torch.npu.config.allow_internal_format = False


def load_text_encoders(class_one, class_two):
    text_encoder_one = class_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", variant=args.variant
    )
    text_encoder_two = class_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", variant=args.variant
    )
    return text_encoder_one, text_encoder_two


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder,
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    prompt,
    max_sequence_length=512,
    device=None,
):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids

    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=False)[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    return prompt_embeds


def _encode_prompt_with_clip(text_encoder, tokenizer, prompt, device):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_overflowing_tokens=False,
        return_length=False,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids

    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=False)

    # Use pooled output of CLIPTextModel
    prompt_embeds = prompt_embeds.pooler_output
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    return prompt_embeds


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="controlnet-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--dataset_file",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) containing the training data of instance images (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--prob_uncond",
        type=float,
        default=0.1,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0.1.",
    )

    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=50)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=5000,
        help=(
            "Save a checkpoint of the training state every X updates. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=10,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant_with_warmup",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=200, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--use_adafactor",
        action="store_true",
        help=(
            "Adafactor is a stochastic optimization method based on Adam that reduces memory usage while retaining"
            "the empirical benefits of adaptivity. This is achieved through maintaining a factored representation "
            "of the squared gradient accumulator across training steps."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--enable_npu_flash_attention", action="store_true", help="Whether or not to use npu flash attention."
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--num_double_layers",
        type=int,
        default=4,
        help="Number of double layers in the controlnet (default: 4).",
    )
    parser.add_argument(
        "--num_single_layers",
        type=int,
        default=1,
        help="Number of single layers in the controlnet (default: 1).",
    )

    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1.0,
        help="the guidance scale used for transformer.",
    )

    parser.add_argument(
        "--save_weight_dtype",
        type=str,
        default="fp32",
        choices=[
            "fp16",
            "bf16",
            "fp32",
        ],
        help=("Preserve precision type according to selected weight"),
    )

    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="logit_normal",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help=('We default to the "none" weighting scheme for uniform sampling and uniform loss'),
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument("--use_time_shift", action="store_true", help="Whether to use dynamic shifting.")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."
        )

    return args


def main(args):
    logging_out_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=str(logging_out_dir))

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        # DEBUG, INFO, WARNING, ERROR, CRITICAL
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if exists(join(args.pretrained_model_name_or_path, "empty_prompt_embeds")):
        empty_prompt_embeds = torch.load(join(args.pretrained_model_name_or_path, "empty_prompt_embeds"))
        prompt_embeds = empty_prompt_embeds["prompt_embeds"].repeat_interleave(args.train_batch_size, dim=0)
        pooled_prompt_embeds = empty_prompt_embeds["pooled_prompt_embeds"].repeat_interleave(args.train_batch_size, dim=0)
        text_ids = empty_prompt_embeds["text_ids"]

        prompt_embeds = prompt_embeds.to(accelerator.device, dtype=weight_dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device, dtype=weight_dtype)
        text_ids = text_ids.to(accelerator.device, dtype=weight_dtype)
    else:
        # Load the tokenizers
        tokenizer_one = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
        )
        tokenizer_two = T5TokenizerFast.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer_2",
        )

        # import correct text encoder classes
        text_encoder_cls_one = import_model_class_from_model_name_or_path(
            args.pretrained_model_name_or_path, subfolder="text_encoder"
        )
        text_encoder_cls_two = import_model_class_from_model_name_or_path(
            args.pretrained_model_name_or_path, subfolder="text_encoder_2"
        )
        text_encoder_one, text_encoder_two = load_text_encoders(text_encoder_cls_one, text_encoder_cls_two)
        text_encoder_one.requires_grad_(False)
        text_encoder_two.requires_grad_(False)
        text_encoder_one.to(accelerator.device, dtype=weight_dtype)
        text_encoder_two.to(accelerator.device, dtype=weight_dtype)

        prompt = ""
        with torch.no_grad():
            pooled_prompt_embeds = _encode_prompt_with_clip(
                text_encoder=text_encoder_one,
                tokenizer=tokenizer_one,
                prompt=prompt,
                device=accelerator.device,
            )

            prompt_embeds = _encode_prompt_with_t5(
                text_encoder=text_encoder_two,
                tokenizer=tokenizer_two,
                prompt=prompt,
                max_sequence_length=args.max_sequence_length,
                device=accelerator.device,
            )

            prompt_embeds = prompt_embeds.repeat_interleave(args.train_batch_size, dim=0)
            pooled_prompt_embeds = pooled_prompt_embeds.repeat_interleave(args.train_batch_size, dim=0)

            text_ids = torch.zeros(prompt_embeds.shape[1], 3)
            text_ids = text_ids.to(device=accelerator.device, dtype=weight_dtype)
            save_tensors = {
                "prompt_embeds": prompt_embeds.cpu(),
                "pooled_prompt_embeds": pooled_prompt_embeds.cpu(),
                "text_ids": text_ids.cpu()
            }
        del tokenizer_one
        del tokenizer_two
        del text_encoder_cls_one
        del text_encoder_cls_two
        free_memory()

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae", variant=args.variant)
    flux_transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer", variant=args.variant)

    # Load controlnet
    if args.controlnet_model_name_or_path:
        logger.info("Loading existing controlnet weights")
        flux_controlnet = FluxControlNetModel.from_pretrained(args.controlnet_model_name_or_path)
    else:
        logger.info("Initializing controlnet weights from transformer")
        # we can define the num_layers, num_single_layers,
        flux_controlnet = FluxControlNetModel.from_transformer(
            flux_transformer,
            attention_head_dim=flux_transformer.config["attention_head_dim"],
            num_attention_heads=flux_transformer.config["num_attention_heads"],
            num_layers=args.num_double_layers,
            num_single_layers=args.num_single_layers,
        )
    vae.requires_grad_(False)
    vae.to(accelerator.device, dtype=weight_dtype)
    flux_transformer.requires_grad_(False)
    flux_transformer.to(accelerator.device, dtype=weight_dtype)

    logger.info("all models loaded successfully")

    # make img_ids
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    latent_image_ids = FluxPipeline._prepare_latent_image_ids(
        args.train_batch_size,
        args.resolution // vae_scale_factor,
        args.resolution // vae_scale_factor,
        accelerator.device,
        weight_dtype,
    )

    if args.use_time_shift:
        # compute mu to shifting
        mu = calculate_shift(
            latent_image_ids.shape[0],
            noise_scheduler.config.base_image_seq_len,
            noise_scheduler.config.max_image_seq_len,
            noise_scheduler.config.base_shift,
            noise_scheduler.config.max_shift,
        )
        sigmas = noise_scheduler.time_shift(mu, 1.0, noise_scheduler.sigmas)
        noise_scheduler.sigmas = sigmas
        timesteps = sigmas * noise_scheduler.config.num_train_timesteps
        noise_scheduler.timesteps = timesteps

    # handle guidance vector
    use_guidance_embeds = flux_transformer.config.guidance_embeds
    if use_guidance_embeds:
        guidance_vec = torch.tensor([args.guidance_scale], device=accelerator.device)
        guidance_vec = guidance_vec.repeat_interleave(args.train_batch_size, dim=0)
    else:
        guidance_vec = None

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    if args.enable_npu_flash_attention:
        if is_torch_npu_available():
            logger.info("npu flash attention enabled.")
            flux_transformer.enable_npu_flash_attention()
        else:
            raise ValueError("npu flash attention requires torch_npu extensions and is supported only on npu devices.")

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            flux_transformer.enable_xformers_memory_efficient_attention()
            flux_controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        flux_transformer.enable_gradient_checkpointing()
        flux_controlnet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    params_to_optimize = list(flux_controlnet.parameters())
    num_params = sum([a.numel() for a in params_to_optimize])
    print(f"Model params: {num_params / 1000 / 1000} M")
    # use adafactor optimizer to save gpu memory
    if args.use_adafactor:
        from transformers import Adafactor

        optimizer = Adafactor(
            params_to_optimize,
            lr=args.learning_rate,
            scale_parameter=False,
            relative_step=False,
            # warmup_init=True,
            weight_decay=args.adam_weight_decay,
        )
    else:
        optimizer = optimizer_class(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    # Dataset and DataLoaders creation:
    train_dataset = ControlNetDataset(
        dataset_file=args.dataset_file,
        img_size=args.resolution,
        prob_uncond=args.prob_uncond,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=flux_collate_fn,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
        pin_memory=True,
    )

    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            args.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
        )
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )
    # Prepare everything with our `accelerator`.
    flux_controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        flux_controlnet, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps * accelerator.num_processes:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader) // args.gradient_accumulation_steps}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(flux_controlnet):
                # Convert images to latent space
                # vae encode
                pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
                pixel_latents_tmp = vae.encode(pixel_values).latent_dist.sample()
                pixel_latents_tmp = (pixel_latents_tmp - vae.config.shift_factor) * vae.config.scaling_factor

                pixel_latents = FluxControlNetPipeline._pack_latents(
                    pixel_latents_tmp,
                    pixel_values.shape[0],
                    pixel_latents_tmp.shape[1],
                    pixel_latents_tmp.shape[2],
                    pixel_latents_tmp.shape[3],
                )

                control_values = batch["conditioning_pixel_values"].to(dtype=weight_dtype)
                control_latents = vae.encode(control_values).latent_dist.sample()
                control_latents = (control_latents - vae.config.shift_factor) * vae.config.scaling_factor
                control_image = FluxControlNetPipeline._pack_latents(
                    control_latents,
                    control_values.shape[0],
                    control_latents.shape[1],
                    control_latents.shape[2],
                    control_latents.shape[3],
                )

                bsz = pixel_latents.shape[0]
                noise = torch.randn_like(pixel_latents).to(accelerator.device).to(dtype=weight_dtype)
                # Sample a random timestep for each image
                # for weighting schemes where we sample timesteps non-uniformly
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler.config.num_train_timesteps).long()
                timesteps = noise_scheduler.timesteps[indices].to(device=pixel_latents.device)

                # Add noise according to flow matching.
                sigmas = get_sigmas(timesteps, n_dim=pixel_latents.ndim, dtype=pixel_latents.dtype)
                noisy_model_input = (1.0 - sigmas) * pixel_latents + sigmas * noise

                controlnet_block_samples, controlnet_single_block_samples = flux_controlnet(
                    hidden_states=noisy_model_input,
                    controlnet_cond=control_image,
                    timestep=timesteps / 1000,
                    guidance=guidance_vec.detach() if use_guidance_embeds else None,
                    pooled_projections=pooled_prompt_embeds.detach(),
                    encoder_hidden_states=prompt_embeds.detach(),
                    txt_ids=text_ids.detach(),
                    img_ids=latent_image_ids.detach(),
                    return_dict=False,
                )

                noise_pred = flux_transformer(
                    hidden_states=noisy_model_input,
                    timestep=timesteps / 1000,
                    guidance=guidance_vec.detach() if use_guidance_embeds else None,
                    pooled_projections=pooled_prompt_embeds.detach(),
                    encoder_hidden_states=prompt_embeds.detach(),
                    controlnet_block_samples=[
                        sample.to(dtype=weight_dtype) for sample in controlnet_block_samples
                    ] if controlnet_block_samples is not None else None,
                    controlnet_single_block_samples=[
                        sample.to(dtype=weight_dtype) for sample in controlnet_single_block_samples
                    ] if controlnet_single_block_samples is not None else None,
                    txt_ids=text_ids.detach(),
                    img_ids=latent_image_ids.detach(),
                    return_dict=False,
                )[0]

                loss = F.mse_loss(noise_pred.float(), (noise - pixel_latents).float(), reduction="mean")
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    params_to_clip = flux_controlnet.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # DeepSpeed requires saving weights on every device; saving weights only on the main process would cause issues.
                # accelerator.distributed_type == DistributedType.DEEPSPEED or
                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = join(args.output_dir, f"checkpoint-{global_step}")
                        if not exists(save_path):
                            os.makedirs(save_path, exist_ok=True)
                        model = unwrap_model(flux_controlnet).to(torch.float32)
                        model.save_pretrained(join(save_path, "flux_controlnet"))
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        flux_controlnet = unwrap_model(flux_controlnet)
        save_weight_dtype = torch.float32
        if args.save_weight_dtype == "fp16":
            save_weight_dtype = torch.float16
        elif args.save_weight_dtype == "bf16":
            save_weight_dtype = torch.bfloat16
        flux_controlnet.to(save_weight_dtype)
        if args.save_weight_dtype != "fp32":
            flux_controlnet.save_pretrained(args.output_dir, variant=args.save_weight_dtype)
        else:
            flux_controlnet.save_pretrained(args.output_dir)

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
