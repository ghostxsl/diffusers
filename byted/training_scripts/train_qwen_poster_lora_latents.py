#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
# Copyright 2025 The ByteDance Inc. CreativeAI team. All rights reserved.
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
from os.path import exists, join
import shutil
from pathlib import Path

import torch
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from tqdm.auto import tqdm
import transformers

import diffusers
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from diffusers.pipelines.flux import FluxPipeline
from diffusers.loaders.lora_pipeline import QwenImageLoraLoaderMixin
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.models.autoencoders import AutoencoderKLQwenImage
from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3, cast_training_params
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.utils.import_utils import is_torch_npu_available
from diffusers.data.dataset import QwenImageDataset
from diffusers.data.utils import flux_kontext_collate_fn


logger = get_logger(__name__)
if is_torch_npu_available():
    torch.npu.config.allow_internal_format = False


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 8192,
    base_shift: float = 0.5,
    max_shift: float = 0.9,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


@torch.no_grad()
def qwen_encode_vae_image(image, vae, latents_mean, latents_std, sample_mode="sample"):
    image = image.unsqueeze(2).to(vae.device, dtype=vae.dtype)
    if sample_mode == "sample":
        image_latents = vae.encode(image).latent_dist.sample()
    elif sample_mode == "argmax":
        image_latents = vae.encode(image).latent_dist.mode()
    else:
        raise Exception("Could not access latents of provided `sample_mode`")

    image_latents = (image_latents - latents_mean) * latents_std

    return image_latents.squeeze(2)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--transformer_model_path",
        type=str,
        default=None,
        help="Path to transformer model.",
    )
    parser.add_argument(
        "--lora_model_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="qwen-lora",
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
        "--cond_size",
        type=str,
        default="1024x1024",
        help=(
            "The resolution for conditional images, all the images in the train dataset will be resized to this"
            " resolution"
        ),
    )

    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=1000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint."
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
        default=1e-4,
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
            ' "constant", "constant_with_warmup", "piecewise_constant"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=100, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--lr_step_rules",
        type=str,
        default="1:10000,0.1:20000,0.01",
        help=(
            "A string representing the step rules to use. This is only used by the `PIECEWISE_CONSTANT` scheduler. "
            "rule_steps='1:10,0.1:20,0.01:30,0.005' it means that the learning rate. if multiple 1 for the first 10 steps, "
            "multiple 0.1 for the next 20 steps, multiple 0.01 for the next 30 steps and multiple 0.005 for the other steps."
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
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )

    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for unet params")
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )
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
        "--rank",
        type=int,
        default=128,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--use_time_shift",
        action="store_true",
        help="Whether to use dynamic shifting."
    )
    parser.add_argument(
        "--use_t2i_pretrained",
        action="store_true",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if len(args.cond_size.split("x")) == 1:
        args.cond_size = [int(args.cond_size), ] * 2
    elif len(args.cond_size.split("x")) == 2:
        args.cond_size = tuple([int(r) for r in args.cond_size.split("x")])
    else:
        raise Exception(f"Error `cond_size` type({type(args.cond_size)}): {args.cond_size}.")

    return args


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
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

    # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Load scheduler and models
    vae = AutoencoderKLQwenImage.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    vae_scale_factor = 2 ** len(vae.temperal_downsample)
    latents_mean = (torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1)).to(accelerator.device)
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(accelerator.device)
    vae.requires_grad_(False)
    vae.to(accelerator.device, dtype=weight_dtype)

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler")

    if args.transformer_model_path:
        transformer = QwenImageTransformer2DModel.from_pretrained(
            args.transformer_model_path, torch_dtype=weight_dtype)
    else:
        transformer = QwenImageTransformer2DModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="transformer", torch_dtype=weight_dtype)
    transformer.requires_grad_(False)
    transformer.to(accelerator.device, dtype=weight_dtype)

    # now we will add new LoRA weights to the attention layers
    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank if args.lora_alpha is None else args.lora_alpha,
        init_lora_weights="gaussian",
        target_modules=[
            "to_q", "to_k", "to_v", "to_out.0",
            "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out",
            "net.0.proj", "net.2",
        ],
    )
    transformer.add_adapter(transformer_lora_config)

    if args.lora_model_path:
        lora_state_dict = QwenImageLoraLoaderMixin.lora_state_dict(args.lora_model_path)
        transformer_state_dict = {
            f'{k.replace("transformer.", "")}': v for k, v in lora_state_dict.items() if k.startswith("transformer.")
        }
        set_peft_model_state_dict(transformer, transformer_state_dict, adapter_name="default")

    cast_training_params(transformer)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
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

    # Optimization parameters
    params_to_optimize = [p for p in transformer.parameters() if p.requires_grad]
    num_params = sum([a.numel() for a in params_to_optimize])
    print(f"Model params: {num_params / 1000 / 1000} M")
    params_to_clip = params_to_optimize
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Dataset and DataLoaders creation:
    train_dataset = QwenImageDataset(
        dataset_file=args.dataset_file,
        t2i=args.use_t2i_pretrained,
        cond_size=args.cond_size,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=flux_kontext_collate_fn,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
        pin_memory=True,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        step_rules=args.lr_step_rules,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
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

    cache_sigmas = {}
    def get_sigmas_and_timesteps_with_time_shift(indices, image_seq_len):
        if image_seq_len in cache_sigmas:
            sigmas = cache_sigmas[image_seq_len][indices]
        else:
            # compute mu to shifting
            mu = calculate_shift(
                image_seq_len,
                noise_scheduler.config.get("base_image_seq_len", 256),
                noise_scheduler.config.get("max_image_seq_len", 8192),
                noise_scheduler.config.get("base_shift", 0.5),
                noise_scheduler.config.get("max_shift", 0.9),
            )
            sigmas = noise_scheduler.time_shift(mu, 1.0, noise_scheduler.sigmas)
            # sigmas = noise_scheduler.stretch_shift_to_terminal(sigmas)
            cache_sigmas[image_seq_len] = sigmas
            sigmas = sigmas[indices]
        timesteps = sigmas * noise_scheduler.config.num_train_timesteps
        return sigmas, timesteps

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    transformer.train()
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer):
                # Convert images to latent space
                model_input = qwen_encode_vae_image(
                    batch["image_latents"], vae, latents_mean, latents_std).to(weight_dtype)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]

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
                if args.use_time_shift:
                    image_seq_len = (model_input.shape[2] // 2) * (model_input.shape[3] // 2)
                    sigmas, timesteps = get_sigmas_and_timesteps_with_time_shift(indices, image_seq_len)
                    sigmas = sigmas.to(accelerator.device, dtype=weight_dtype)
                    timesteps = timesteps.to(accelerator.device)
                else:
                    timesteps = noise_scheduler.timesteps[indices].to(device=model_input.device)
                    sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)

                # Add noise according to flow matching.
                # zt = (1 - texp) * x + texp * z1
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

                cond_model_input = qwen_encode_vae_image(
                    batch["cond_image_latents"],
                    vae,
                    latents_mean,
                    latents_std,
                    sample_mode="argmax",
                ).to(weight_dtype)

                img_shapes = [
                    [
                        (1, model_input.shape[2] // 2, model_input.shape[3] // 2),
                        (1, cond_model_input.shape[2] // 2, cond_model_input.shape[3] // 2)
                    ]
                ]

                packed_noisy_model_input = FluxPipeline._pack_latents(
                    noisy_model_input, *noisy_model_input.shape)
                orig_inp_shape = packed_noisy_model_input.shape

                packed_cond_model_input = FluxPipeline._pack_latents(
                    cond_model_input, *cond_model_input.shape)
                packed_noisy_model_input = torch.cat([packed_noisy_model_input, packed_cond_model_input], dim=1)

                prompt_embeds = batch["prompt_embeds"]
                prompt_embeds_mask = batch["prompt_embeds_mask"]

                # Predict the noise residual
                model_pred = transformer(
                    hidden_states=packed_noisy_model_input,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_mask=prompt_embeds_mask,
                    timestep=timesteps / 1000,
                    img_shapes=img_shapes,
                    txt_seq_lens=prompt_embeds_mask.sum(dim=1).tolist(),
                    return_dict=False,
                )[0]
                model_pred = model_pred[:, : orig_inp_shape[1]]

                model_pred = FluxPipeline._unpack_latents(
                    model_pred,
                    height=int(model_input.shape[2] * vae_scale_factor),
                    width=int(model_input.shape[3] * vae_scale_factor),
                    vae_scale_factor=vae_scale_factor,
                )

                # these weighting schemes use a uniform timestep sampling
                # and instead post-weight the loss
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

                # flow matching loss
                target = noise - model_input

                # Compute regular loss.
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

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

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        if not exists(save_path):
                            os.makedirs(save_path, exist_ok=True)

                        model = unwrap_model(transformer)
                        transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                        QwenImageLoraLoaderMixin.save_lora_weights(
                            save_path,
                            transformer_lora_layers=transformer_lora_layers_to_save,
                        )
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Save the weights
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        transformer = unwrap_model(transformer)
        transformer_lora_layers_to_save = get_peft_model_state_dict(transformer)
        QwenImageLoraLoaderMixin.save_lora_weights(
            join(args.output_dir),
            transformer_lora_layers=transformer_lora_layers_to_save,
        )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
