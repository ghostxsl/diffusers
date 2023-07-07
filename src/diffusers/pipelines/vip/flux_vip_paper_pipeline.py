# Copyright 2024 Black Forest Labs and The HuggingFace Team. All rights reserved.
# Copyright 2025 The VIP Inc. AIGC team. All rights reserved.
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
# limitations under the License.

import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from PIL import Image
import torch
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
)

from ...image_processor import PipelineImageInput, VaeImageProcessor
from ...loaders import FluxLoraLoaderMixin
from ...models.autoencoders import AutoencoderKL
from ...models.transformers import FluxTransformer2DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import (
    USE_PEFT_BACKEND,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline
from diffusers.models.controlnets.controlnet_flux import FluxControlNetModel
from diffusers.models.vip.controlnetxs_flux import FluxControlNetXSModel


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Copied from diffusers.pipelines.flux.pipeline_flux.calculate_shift
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class FluxVIPPaperPipeline(DiffusionPipeline, FluxLoraLoaderMixin):

    model_cpu_offload_seq = "transformer->vae"
    _optional_components = ["image_encoder", "feature_extractor", "controlnet",]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        transformer: FluxTransformer2DModel,
        image_encoder: CLIPVisionModelWithProjection = None,
        feature_extractor: CLIPImageProcessor = None,
        controlnet: FluxControlNetModel = None,
        controlnetxs: FluxControlNetXSModel = None,
        fill_ic=False,
    ):
        super().__init__()

        self.register_modules(
            scheduler=scheduler,
            vae=vae,
            transformer=transformer,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
            controlnet=controlnet,
            controlnetxs=controlnetxs,
        )
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels)) if hasattr(self, "vae") and self.vae is not None else 16
        )
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length if hasattr(self, "tokenizer") and self.tokenizer is not None else 77
        )
        self.default_sample_size = 64
        self.fill_ic = fill_ic

    # Copied from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3_inpaint.StableDiffusion3InpaintPipeline._encode_vae_image
    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i])
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = retrieve_latents(self.vae.encode(image), generator=generator)

        image_latents = (image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        return image_latents

    # Copied from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3_img2img.StableDiffusion3Img2ImgPipeline.get_timesteps
    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(num_inference_steps * strength, num_inference_steps)

        t_start = int(max(num_inference_steps - init_timestep, 0))
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(t_start * self.scheduler.order)

        return timesteps, num_inference_steps - t_start

    @staticmethod
    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline._prepare_latent_image_ids
    def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
        latent_image_ids = torch.zeros(height // 2, width // 2, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )

        return latent_image_ids.to(device=device, dtype=dtype)

    @staticmethod
    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline._pack_latents
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

        return latents

    @staticmethod
    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline._unpack_latents
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape

        height = height // vae_scale_factor
        width = width // vae_scale_factor

        latents = latents.view(batch_size, height, width, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // (2 * 2), height * 2, width * 2)

        return latents

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        image_latents=None,
        timestep=None,
        latents=None,
    ):
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        height = 2 * (int(height) // self.vae_scale_factor)
        width = 2 * (int(width) // self.vae_scale_factor)

        shape = (batch_size, num_channels_latents, height, width)
        latent_image_ids = self._prepare_latent_image_ids(batch_size, height, width, device, dtype)

        if latents is not None:
            return latents.to(device=device, dtype=dtype), latent_image_ids

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        if image_latents is not None:
            latents = self.scheduler.scale_noise(image_latents, timestep, latents)
        latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)
        return latents, latent_image_ids,

    def prepare_ip_adapter_image_embeds(
            self,
            image,
            device,
            num_images_per_prompt,
            output_hidden_states=True,
    ):
        if image is None:
            image = Image.new("RGB", (448, 448))

        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor(image, return_tensors="pt").pixel_values

        dtype = next(self.image_encoder.parameters()).dtype
        image = image.to(device=device, dtype=dtype)
        if output_hidden_states:
            image_enc_hidden_states = self.image_encoder(
                image, output_hidden_states=True).hidden_states[-2]
            image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(
                num_images_per_prompt, dim=0)
            uncond_image_enc_hidden_states = torch.zeros_like(image_enc_hidden_states)

            return image_enc_hidden_states, uncond_image_enc_hidden_states
        else:
            image_embeds = self.image_encoder(image).image_embeds
            image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            uncond_image_embeds = torch.zeros_like(image_embeds)

            return image_embeds, uncond_image_embeds

    @property
    def guidance_vector(self):
        return self._guidance_vector

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    def __call__(
        self,
        image: PipelineImageInput = None,
        tgt_image: PipelineImageInput = None,
        control_image: PipelineImageInput = None,
        prompt_embeds=None,
        pooled_prompt_embeds=None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        strength: float = 1.0,
        num_inference_steps: int = 35,
        timesteps: List[int] = None,
        guidance_vector: float = 1.0,
        guidance_scale: float = 1.0,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        max_sequence_length: int = 512,
        **kwargs
    ):
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        if self.fill_ic:
            img_width = width * 2

        self._guidance_vector = guidance_vector
        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Preprocess image
        init_image = self.image_processor.preprocess(image, height=height, width=width)
        init_image = init_image.to(device=device, dtype=prompt_embeds.dtype)
        if self.fill_ic:
            image_latents = self._encode_vae_image(image=init_image, generator=generator)
            masked_image = torch.cat([image_latents, torch.zeros_like(image_latents)], dim=-1)
            masked_image = self._pack_latents(
                masked_image,
                batch_size=masked_image.shape[0],
                num_channels_latents=masked_image.shape[1],
                height=masked_image.shape[2],
                width=masked_image.shape[3],
            ).to(device=device, dtype=prompt_embeds.dtype)
            if tgt_image is not None:
                tgt_image = self.image_processor.preprocess(tgt_image, height=height, width=width)
                tgt_image = tgt_image.to(device=device, dtype=prompt_embeds.dtype)
                tgt_image = self._encode_vae_image(image=tgt_image, generator=generator)
                image_latents = torch.cat([image_latents, tgt_image], dim=3)

        if self.image_encoder is not None:
            image_embeds, negative_image_embeds = self.prepare_ip_adapter_image_embeds(
                image,
                device,
                num_images_per_prompt,
            )

        if self.fill_ic:
            vae_scale_factor = self.vae_scale_factor // 2
            latent_h = int(height) // vae_scale_factor
            latent_w = int(width) // vae_scale_factor

            mask = torch.cat([
                torch.zeros([num_images_per_prompt, height, width]),
                torch.ones([num_images_per_prompt, height, width])], dim=-1)
            mask = mask.view(
                num_images_per_prompt, latent_h, vae_scale_factor, latent_w * 2, vae_scale_factor
            )  # batch_size, height, 8, width, 8
            mask = mask.permute(0, 2, 4, 1, 3)  # batch_size, 8, 8, height, width
            mask = mask.reshape(
                num_images_per_prompt, vae_scale_factor * vae_scale_factor, latent_h, latent_w * 2
            )  # batch_size, 8*8, height, width
            packed_mask = self._pack_latents(
                mask,
                batch_size=mask.shape[0],
                num_channels_latents=mask.shape[1],
                height=mask.shape[2],
                width=mask.shape[3],
            ).to(device=device, dtype=prompt_embeds.dtype)

        # 4.Prepare timesteps
        if self.fill_ic:
            image_seq_len = (int(height) // self.vae_scale_factor) * (int(img_width) // self.vae_scale_factor)
        else:
            image_seq_len = (int(height) // self.vae_scale_factor) * (int(width) // self.vae_scale_factor)
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        if kwargs.get("clip_mu", True):
            mu = min(self.scheduler.config.max_shift, mu)
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas,
            mu=mu,
        )
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)

        if num_inference_steps < 1:
            raise ValueError(
                f"After adjusting the num_inference_steps by strength parameter: {strength}, the number of pipeline"
                f"steps is {num_inference_steps} which is < 1 and not appropriate for this pipeline."
            )
        latent_timestep = timesteps[:1].repeat(num_images_per_prompt)

        # 5. Prepare latent variables
        num_channels_latents = self.vae.config.latent_channels
        latents, latent_image_ids = self.prepare_latents(
            num_images_per_prompt,
            num_channels_latents,
            height,
            img_width if self.fill_ic else width,
            prompt_embeds.dtype,
            device,
            generator,
            image_latents=image_latents if tgt_image is not None else None,
            timestep=latent_timestep,
        )
        text_ids = torch.zeros(max_sequence_length, 3).to(device=device, dtype=prompt_embeds.dtype)

        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_vector, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        control_image = self.image_processor.preprocess(control_image, height=height, width=width)
        control_image = control_image.to(device, dtype=prompt_embeds.dtype)

        if self.controlnetxs is None:
            control_image = self._encode_vae_image(image=control_image, generator=generator)
            if self.fill_ic:
                control_image = torch.cat([torch.zeros_like(control_image), control_image], dim=-1)
            control_image = self._pack_latents(
                control_image, num_images_per_prompt,
                control_image.shape[1],
                control_image.shape[2],
                control_image.shape[3])

        if self.fill_ic:
            condition_latents = torch.cat([masked_image, packed_mask, control_image], dim=-1)
        else:
            condition_latents = control_image

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                if self.controlnetxs is not None:
                    noise_pred = self.controlnetxs(
                        base_model=self.transformer,
                        hidden_states=latents,
                        controlnet_cond=condition_latents,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_prompt_embeds,
                        timestep=timestep,
                        img_ids=latent_image_ids,
                        txt_ids=text_ids,
                        guidance=guidance,
                        image_embeds=image_embeds if self.image_encoder is not None else None,
                        return_dict=False,
                    )[0]
                elif self.controlnet is not None:
                    controlnet_block_samples, controlnet_single_block_samples = self.controlnet(
                        hidden_states=latents,
                        controlnet_cond=condition_latents,
                        timestep=timestep,
                        guidance=guidance,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        return_dict=False,
                    )

                    noise_pred = self.transformer(
                        hidden_states=latents,
                        timestep=timestep,
                        guidance=guidance,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_prompt_embeds,
                        controlnet_block_samples=[
                            sample.to(dtype=latents.dtype) for sample in controlnet_block_samples
                        ] if controlnet_block_samples is not None else None,
                        controlnet_single_block_samples=[
                            sample.to(dtype=latents.dtype) for sample in controlnet_single_block_samples
                        ] if controlnet_single_block_samples is not None else None,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        image_embeds=image_embeds if self.image_encoder is not None else None,
                        return_dict=False,
                    )[0]
                else:
                    noise_pred = self.transformer(
                        hidden_states=torch.cat([latents, condition_latents], dim=-1),
                        timestep=timestep,
                        guidance=guidance,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        image_embeds=image_embeds if self.image_encoder is not None else None,
                        return_dict=False,
                    )[0]

                if do_classifier_free_guidance:
                    noise_pred_uncond = self.transformer(
                        hidden_states=torch.cat([latents, control_image], dim=-1),
                        timestep=timestep,
                        guidance=guidance,
                        encoder_hidden_states=negative_prompt_embeds,
                        pooled_projections=negative_pooled_prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        image_embeds=negative_image_embeds if self.image_encoder is not None else None,
                        return_dict=False,
                    )[0]
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        if self.fill_ic:
            latents = self._unpack_latents(latents, height, img_width, self.vae_scale_factor)
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type="pil")

        # Offload all models
        self.maybe_free_model_hooks()

        return image
