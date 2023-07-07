# Copyright 2024 The HuggingFace Team. All rights reserved.
# Copyright 2024 The VIP AIGC Team. All rights reserved.
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
import torch
from torchvision.transforms import ToTensor, Normalize, Compose
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor, CLIPVisionModelWithProjection

from ...image_processor import VaeImageProcessor
from ...loaders import LoraLoaderMixin, TextualInversionLoaderMixin, IPAdapterMixin
from ...models import AutoencoderKL, UNet2DConditionModel, ControlNetModel
from ...schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from ...utils import USE_PEFT_BACKEND, logging, get_promt_embedding
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline
from diffusers.models.reference_attention import ReferenceAttentionControl
from diffusers.models.referencenet import ReferenceNetModel


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import MotionAdapter, AnimateDiffPipeline, DDIMScheduler
        >>> from diffusers.utils import export_to_gif

        >>> adapter = MotionAdapter.from_pretrained("diffusers/motion-adapter")
        >>> pipe = AnimateDiffPipeline.from_pretrained("frankjoshua/toonyou_beta6", motion_adapter=adapter)
        >>> pipe.scheduler = DDIMScheduler(beta_schedule="linear", steps_offset=1, clip_sample=False)
        >>> output = pipe(prompt="A corgi walking in the park")
        >>> frames = output.frames[0]
        >>> export_to_gif(frames, "animation.gif")
        ```
"""


class VIPIPAPoseTransferPipeline(DiffusionPipeline, TextualInversionLoaderMixin, LoraLoaderMixin, IPAdapterMixin):
    model_cpu_offload_seq = "text_encoder->image_encoder->unet->vae"

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
        feature_extractor: CLIPImageProcessor,
        image_encoder: CLIPVisionModelWithProjection,
        controlnet: ControlNetModel,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
            controlnet=controlnet,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.control_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False
        )
        self.normalize_ = Compose([
            ToTensor(),
            Normalize([0.5], [0.5], inplace=True),
        ])

    def encode_image(self, image, device, num_images_per_prompt, do_classifier_free_guidance=False):
        dtype = next(self.image_encoder.parameters()).dtype

        if image is None:
            image = Image.new("RGB", (224, 224))
        image = self.feature_extractor(images=image, return_tensors="pt").pixel_values

        image = image.to(device=device, dtype=dtype)
        image_embeds = self.image_encoder(image, output_hidden_states=True).hidden_states[-2]
        image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)

        uncond_image_embeds = None
        if do_classifier_free_guidance:
            uncond_image_embeds = torch.zeros_like(image_embeds)

        return image_embeds, uncond_image_embeds

    def encode_prompt(
        self,
        prompt,
        negative_prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        padding_prompt: bool = True,
    ):
        prompt_embeds = get_promt_embedding(prompt, self.tokenizer, self.text_encoder, device)

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)
        pos_len = prompt_embeds.shape[1]
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(num_images_per_prompt, pos_len, -1)

        # get unconditional embeddings for classifier free guidance
        negative_prompt_embeds = None
        if do_classifier_free_guidance:
            negative_prompt_embeds = get_promt_embedding(
                negative_prompt, self.tokenizer, self.text_encoder, device)

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)
            neg_len = negative_prompt_embeds.shape[1]
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(num_images_per_prompt, neg_len, -1)

            if padding_prompt:
                # pad embedding
                diff_len = abs(pos_len - neg_len)
                if pos_len > neg_len:
                    pad_embed = negative_prompt_embeds[:, -1].unsqueeze(1).repeat(1, diff_len, 1)
                    negative_prompt_embeds = torch.cat([negative_prompt_embeds, pad_embed], dim=1)
                elif neg_len > pos_len:
                    pad_embed = prompt_embeds[:, -1].unsqueeze(1).repeat(1, diff_len, 1)
                    prompt_embeds = torch.cat([prompt_embeds, pad_embed], dim=1)

        return negative_prompt_embeds, prompt_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_vae_slicing
    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.disable_vae_slicing
    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_vae_tiling
    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        self.vae.enable_tiling()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.disable_vae_tiling
    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_tiling()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_freeu
    def enable_freeu(self, s1: float, s2: float, b1: float, b2: float):
        r"""Enables the FreeU mechanism as in https://arxiv.org/abs/2309.11497.

        The suffixes after the scaling factors represent the stages where they are being applied.

        Please refer to the [official repository](https://github.com/ChenyangSi/FreeU) for combinations of the values
        that are known to work well for different pipelines such as Stable Diffusion v1, v2, and Stable Diffusion XL.

        Args:
            s1 (`float`):
                Scaling factor for stage 1 to attenuate the contributions of the skip features. This is done to
                mitigate "oversmoothing effect" in the enhanced denoising process.
            s2 (`float`):
                Scaling factor for stage 2 to attenuate the contributions of the skip features. This is done to
                mitigate "oversmoothing effect" in the enhanced denoising process.
            b1 (`float`): Scaling factor for stage 1 to amplify the contributions of backbone features.
            b2 (`float`): Scaling factor for stage 2 to amplify the contributions of backbone features.
        """
        if not hasattr(self, "unet"):
            raise ValueError("The pipeline must have `unet` for using FreeU.")
        self.unet.enable_freeu(s1=s1, s2=s2, b1=b1, b2=b2)

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.disable_freeu
    def disable_freeu(self):
        """Disables the FreeU mechanism if enabled."""
        self.unet.disable_freeu()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.StableDiffusionImg2ImgPipeline.get_timesteps
    def get_timesteps(self, num_inference_steps, strength):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order:]

        return timesteps, num_inference_steps - t_start

    def prepare_latents(
            self,
            image,
            batch_size,
            num_channels_latents,
            height,
            width,
            dtype,
            device,
            generator,
            timestep,
            is_strength_max=True,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        image = image.to(device=device, dtype=dtype)
        image_latents = self._encode_vae_image(image=image, generator=generator)

        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = noise if is_strength_max else self.scheduler.add_noise(image_latents, noise, timestep)
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma if is_strength_max else latents

        return latents

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint.StableDiffusionInpaintPipeline._encode_vae_image
    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        if isinstance(generator, list):
            image_latents = [
                self.vae.encode(image[i: i + 1]).latent_dist.sample(generator=generator[i])
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = self.vae.encode(image).latent_dist.sample(generator=generator)

        image_latents = self.vae.config.scaling_factor * image_latents

        return image_latents

    # Copied from diffusers.pipelines.controlnet.pipeline_controlnet.StableDiffusionControlNetPipeline.prepare_image
    def prepare_control_image(
        self,
        image,
        width,
        height,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance=False,
    ):
        image = self.control_image_processor.preprocess(
            image, height=height, width=width).to(dtype=torch.float32)
        image = image.repeat_interleave(num_images_per_prompt, dim=0)
        image = image.to(device=device, dtype=dtype)

        if do_classifier_free_guidance:
            image = torch.cat([image] * 2)

        return image

    @torch.no_grad()
    def __call__(
        self,
        prompt: str = "",
        negative_prompt: str = "",
        image: Image.Image = None,
        control_image: Image.Image = None,
        ip_adapter_image: Optional[Image.Image] = None,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        strength: float = 1.0,
        num_inference_steps: int = 25,
        guidance_scale: float = 4.5,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        cond_scale: float = 1.0,
        output_type: Optional[str] = "pil",
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = {},
        **kwargs
    ):
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 1. Encode input prompt
        negative_prompt_embeds, prompt_embeds = self.encode_prompt(
            prompt,
            negative_prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # 2. Prepare ip_adapter image embedding
        image_embeds, uncond_image_embeds = self.encode_image(
            ip_adapter_image,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance
        )
        if do_classifier_free_guidance:
            image_embeds = torch.cat([uncond_image_embeds, image_embeds])
        if ip_adapter_image is None:
            image_embeds = torch.zeros_like(image_embeds)
        dtype = image_embeds.dtype

        # 3. Prepare control image
        control_image = self.prepare_control_image(
            control_image,
            width,
            height,
            num_images_per_prompt,
            device,
            dtype,
            do_classifier_free_guidance
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = self.get_timesteps(
            num_inference_steps=num_inference_steps, strength=strength
        )
        # at which timestep to set the initial noise (n.b. 50% if strength is 0.5)
        latent_timestep = timesteps[:1].repeat(num_images_per_prompt)
        # create a boolean to check if the strength is set to 1. if so then initialise the latents with pure noise
        is_strength_max = strength == 1.0

        # 5. Prepare latent variables
        image = image.resize((width, height), Image.LANCZOS)
        image = self.normalize_(image).unsqueeze(0)

        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            image,
            num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            dtype,
            device,
            generator,
            latent_timestep,
            is_strength_max=is_strength_max,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.controlnet(
                    base_model=self.unet,
                    sample=latent_model_input,
                    timestep=t,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs={'image_embeds': [image_embeds.unsqueeze(1)]},
                    controlnet_cond=control_image,
                    conditioning_scale=cond_scale,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # Post-processing
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        do_denormalize = [True] * image.shape[0]
        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # remove cache variables
        if getattr(self.scheduler, 'pred_ori_sample_prev', None) is not None:
            delattr(self.scheduler, 'pred_ori_sample_prev')

        # Offload all models
        self.maybe_free_model_hooks()

        return image
