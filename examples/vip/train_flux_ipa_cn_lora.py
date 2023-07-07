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
from os.path import exists, join
import shutil
from pathlib import Path

import torch
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import transformers
from transformers import (
    CLIPTokenizer,
    T5TokenizerFast,
    CLIPTextModel,
    T5EncoderModel,
    CLIPImageProcessor,
    CLIPVisionModelWithProjection
)

import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxPipeline,
)
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from diffusers.models.vip.transformer_flux import FluxTransformer2DModel
from diffusers.models.vip.pose_encoder import PoseEncoder
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3, cast_training_params
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.utils.import_utils import is_torch_npu_available
from diffusers.data import FluxPTDataset, flux_collate_fn
from diffusers.pipelines.flux.pipeline_flux import calculate_shift


logger = get_logger(__name__)
if is_torch_npu_available():
    torch.npu.config.allow_internal_format = False


def get_ipa_and_lora_params(transformer):
    ipa_params = []
    lora_params = []
    for k, v in transformer.named_parameters():
        if "processor" in k or k.startswith("ip_image_proj"):
            ipa_params.append(v)
        elif "lora" in k:
            lora_params.append(v)
    return ipa_params, lora_params


def save_models(output_dir, model):
    os.makedirs(output_dir, exist_ok=True)

    # save controlnet
    model.pose_enc.save_pretrained(join(output_dir, "pose_encoder"))

    # save ipa
    ip_adapter_state_dict = {}
    for k, v in model.transformer.state_dict().items():
        if "processor" in k or k.startswith("ip_image_proj"):
            ip_adapter_state_dict[k] = v
    torch.save(ip_adapter_state_dict, join(output_dir, "ip_adapter_plus.bin"))
    logger.info(f"Saved ipa weight to {output_dir}.")

    # save lora
    transformer_lora_layers_to_save = get_peft_model_state_dict(model.transformer)
    FluxPipeline.save_lora_weights(
        output_dir,
        transformer_lora_layers=transformer_lora_layers_to_save,
    )


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
        "--ip_adapter_model_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--lora_model_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--pose_encoder_model_path",
        type=str,
        default=None,
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
             " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--image_encoder_model_path",
        type=str,
        default=None,
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
        default="flux_pt",
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
        default=1024,
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
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
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
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )

    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1.0,
        help="the FLUX.1 dev variant is a guidance distilled model",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=256,
        help="Maximum sequence length to use with with the T5 text encoder",
    )

    parser.add_argument(
        "--use_time_shift",
        action="store_true",
        help="Whether to use dynamic shifting."
    )
    parser.add_argument(
        "--use_empty_prompt",
        action="store_true",
        help="Whether to use empty prompt."
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=32,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=None,
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


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


class UnionModel(torch.nn.Module):
    def __init__(self, transformer, pose_enc):
        super().__init__()
        self.transformer = transformer
        self.pose_enc = pose_enc

    def forward(
            self,
            hidden_states,
            timestep,
            encoder_hidden_states,
            pooled_projections,
            image_embeds,
            controlnet_cond,
            img_ids,
            txt_ids,
            guidance,
            uncond,
            joint_attention_kwargs=None,
            **kwargs):
        controlnet_cond = self.pose_enc(controlnet_cond, timestep)

        noise_pred = self.transformer(
            hidden_states=hidden_states,
            controlnet_cond=controlnet_cond,
            conditioning_scale=uncond,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            timestep=timestep,
            image_embeds=image_embeds,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guidance,
            joint_attention_kwargs=joint_attention_kwargs,
            return_dict=False,
        )

        return noise_pred


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    # kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
        # kwargs_handlers=[kwargs],
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

    if args.use_empty_prompt:
        assert exists(join(args.pretrained_model_name_or_path, "empty_prompt_embeds"))
        empty_prompt_embeds = torch.load(join(args.pretrained_model_name_or_path, "empty_prompt_embeds"))

        if args.max_sequence_length == 512:
            prompt_embeds = empty_prompt_embeds["prompt_embeds_512"].repeat_interleave(args.train_batch_size, dim=0)
        elif args.max_sequence_length == 256:
            prompt_embeds = empty_prompt_embeds["prompt_embeds_256"].repeat_interleave(args.train_batch_size, dim=0)
        elif args.max_sequence_length == 128:
            prompt_embeds = empty_prompt_embeds["prompt_embeds_128"].repeat_interleave(args.train_batch_size, dim=0)
        else:
            raise Exception(f"No empty_prompt len: {args.max_sequence_length}.")

        pooled_prompt_embeds = empty_prompt_embeds["pooled_prompt_embeds"].repeat_interleave(args.train_batch_size, dim=0)

        prompt_embeds = prompt_embeds.to(accelerator.device, dtype=weight_dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device, dtype=weight_dtype)
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
        text_encoder_one = CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", variant=args.variant
        )
        text_encoder_two = T5EncoderModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder_2", variant=args.variant
        )
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

    # Load the image encoder
    if args.image_encoder_model_path:
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_model_path)
        feature_extractor = CLIPImageProcessor.from_pretrained(args.image_encoder_model_path)
    else:
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            args.pretrained_model_path, subfolder="image_encoder")
        feature_extractor = CLIPImageProcessor.from_pretrained(
            args.pretrained_model_path, subfolder="feature_extractor")
    image_encoder.requires_grad_(False)
    image_encoder.to(accelerator.device, dtype=weight_dtype)

    # Load scheduler and models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        variant=args.variant,
    )
    transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", variant=args.variant
    )

    vae.requires_grad_(False)
    vae.to(accelerator.device, dtype=weight_dtype)
    transformer.requires_grad_(False)

    # Load lora weights
    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank if args.lora_alpha is None else args.lora_alpha,
        init_lora_weights="gaussian",
        target_modules=[
            "to_q", "to_k", "to_v", "to_out.0", "proj_mlp"
        ],
    )
    transformer.add_adapter(transformer_lora_config)
    if args.lora_model_path:
        lora_state_dict = FluxPipeline.lora_state_dict(args.lora_model_path)
        transformer_state_dict = {
            f'{k.replace("transformer.", "")}': v for k, v in lora_state_dict.items() if k.startswith("transformer.")
        }
        set_peft_model_state_dict(transformer, transformer_state_dict, adapter_name="default")

    # Load IP Adapter
    ip_adapter_state_dict = None
    if args.ip_adapter_model_path is not None:
        ip_adapter_state_dict = torch.load(args.ip_adapter_model_path, map_location='cpu')
        # ip_adapter_state_dict.pop("ip_image_proj.latents")
    transformer._init_ip_adapter_plus(
        num_image_text_embeds=64,
        embed_dims=image_encoder.config.hidden_size,
        num_attention_heads=image_encoder.config.num_attention_heads,
        state_dict=ip_adapter_state_dict,
    )

    # Load pose encoder
    pose_enc = PoseEncoder(transformer.inner_dim)
    if args.pose_encoder_model_path:
        logger.info("Loading existing pose encoder weights")
        pose_enc = PoseEncoder.from_pretrained(args.pose_encoder_model_path)

    cast_training_params(transformer)

    # make img_ids
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    latent_hw = args.resolution // vae_scale_factor
    latent_image_ids = FluxPipeline._prepare_latent_image_ids(
        args.train_batch_size,
        latent_hw // 2,
        latent_hw // 2,
        accelerator.device,
        weight_dtype,
    )

    text_ids = torch.zeros(args.max_sequence_length, 3)
    text_ids = text_ids.to(device=accelerator.device, dtype=weight_dtype)

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
    use_guidance_embeds = transformer.config.guidance_embeds
    if use_guidance_embeds:
        guidance_vec = torch.tensor([args.guidance_scale], device=accelerator.device)
        guidance_vec = guidance_vec.repeat_interleave(args.train_batch_size, dim=0)
    else:
        guidance_vec = None

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
    ipa_params, lora_params = get_ipa_and_lora_params(transformer)
    num_params = sum([a.numel() for a in ipa_params])
    print(f"IPA params: {num_params / 1000 / 1000} M")
    num_params = sum([a.numel() for a in lora_params])
    print(f"Lora params: {num_params / 1000 / 1000} M")
    num_params = sum([a.numel() for a in list(pose_enc.parameters())])
    print(f"Pose encoder params: {num_params / 1000 / 1000} M")
    params_to_clip = ipa_params + lora_params + list(pose_enc.parameters())
    # params_to_optimize = [
    #     {"params": ipa_params, "lr": 5e-6},
    #     {"params": lora_params},
    #     {"params": list(pose_enc.parameters())}  #
    # ]
    optimizer = optimizer_class(
        params_to_clip,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Dataset and DataLoaders creation:
    train_dataset = FluxPTDataset(
        dataset_file=args.dataset_file,
        clip_processor=feature_extractor,
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
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    union_model = UnionModel(transformer, pose_enc)
    union_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        union_model, optimizer, train_dataloader, lr_scheduler
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
            with accelerator.accumulate(transformer):
                pixel_values = batch["pixel_values"].to(dtype=vae.dtype)

                # Convert images to latent space
                model_input = vae.encode(pixel_values).latent_dist.sample()
                model_input = (model_input - vae.config.shift_factor) * vae.config.scaling_factor
                model_input = model_input.to(dtype=weight_dtype)

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
                timesteps = noise_scheduler.timesteps[indices].to(device=model_input.device)

                # Add noise according to flow matching.
                # zt = (1 - texp) * x + texp * z1
                sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

                packed_noisy_model_input = FluxPipeline._pack_latents(
                    noisy_model_input,
                    batch_size=model_input.shape[0],
                    num_channels_latents=model_input.shape[1],
                    height=model_input.shape[2],
                    width=model_input.shape[3],
                )

                image_embeds = image_encoder(
                    batch["reference_image"].to(dtype=weight_dtype),
                    output_hidden_states=True).hidden_states[-2]
                image_embeds = image_embeds * batch["uncond"].to(dtype=weight_dtype)

                control_image = batch["conditioning_pixel_values"].to(dtype=weight_dtype)

                # Predict the noise residual
                model_pred = union_model(
                    hidden_states=packed_noisy_model_input,
                    timestep=timesteps,
                    guidance=guidance_vec.detach() if use_guidance_embeds else None,
                    encoder_hidden_states=prompt_embeds.detach(),
                    pooled_projections=pooled_prompt_embeds.detach(),
                    txt_ids=text_ids.detach(),
                    img_ids=latent_image_ids.detach(),
                    image_embeds=image_embeds,
                    controlnet_cond=control_image,
                    uncond=batch["uncond"].to(dtype=weight_dtype),
                    return_dict=False,
                )[0]

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

                # accelerator.distributed_type == DistributedType.DEEPSPEED or
                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if accelerator.is_main_process and args.checkpoints_total_limit is not None:
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
                        save_models(save_path, unwrap_model(union_model))
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Save the weights
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_models(args.output_dir, unwrap_model(union_model))

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
