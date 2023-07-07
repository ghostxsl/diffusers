# Copyright 2023 The HuggingFace Team. All rights reserved.
# Copyright 2023 The VIP AIGC Team. All rights reserved.
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
from torchvision.transforms import Compose, ToTensor, Normalize
from PIL import Image

from ...image_processor import VaeImageProcessor
from ...loaders import LoraLoaderMixin, TextualInversionLoaderMixin, IPAdapterMixin
from diffusers.models.vip import VIPUNet2DConditionModel
from ...schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from ...utils import USE_PEFT_BACKEND, logging
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline
from diffusers.models.reference_attention import ReferenceAttentionControl
from diffusers.models.vip import ReferenceTextureModel


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


class VIPPixelPoseTransferPipeline(DiffusionPipeline, TextualInversionLoaderMixin, LoraLoaderMixin, IPAdapterMixin):
    model_cpu_offload_seq = "unet->referencenet"

    def __init__(
        self,
        unet: VIPUNet2DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
        referencenet: ReferenceTextureModel,
    ):
        super().__init__()

        self.register_modules(
            unet=unet,
            scheduler=scheduler,
            referencenet=referencenet,
        )
        self.control_image_processor = VaeImageProcessor(
            do_convert_rgb=True, do_normalize=False
        )

        self.reference_control_reader = ReferenceAttentionControl(
            self.unet, fusion_blocks='full', hack_type='vip')

        self.normalize_image = Compose(
            [
                ToTensor(),
                Normalize([0.5], [0.5], inplace=True),
            ]
        )

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

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.check_inputs
    def check_inputs(
        self,
        prompt,
        height,
        width,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    # Copied from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_synth.TextToVideoSDPipeline.prepare_latents
    def prepare_latents(
        self, batch_size, num_channels_latents, height, width, dtype, device, generator
    ):
        shape = (
            batch_size,
            num_channels_latents,
            height,
            width,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

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

    def decode_images(self, latents):
        images = (latents / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).float().numpy()

        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    @torch.no_grad()
    def __call__(
        self,
        reference_image: Image.Image = None,
        control_image: Image.Image = None,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: int = 25,
        guidance_scale: float = 6.5,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "pil",
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = {},
        **kwargs,
    ):
        device = self._execution_device
        dtype = self.unet.dtype
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 1. Prepare image embedding and reference latents
        reference_image = reference_image.resize((width, height), Image.LANCZOS)
        reference_image = self.normalize_image(reference_image).unsqueeze(0)

        if do_classifier_free_guidance:
            uncond_image = torch.zeros_like(reference_image)
        reference_latents = torch.cat([uncond_image, reference_image]) if do_classifier_free_guidance else reference_image
        reference_latents = reference_latents.to(device=device, dtype=dtype)

        # 2. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 3. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            dtype,
            device,
            generator,
        )

        # 4. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # Prepare control image
        control_image = self.prepare_control_image(
            control_image,
            width,
            height,
            num_images_per_prompt,
            device,
            dtype,
            do_classifier_free_guidance
        )

        # Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                self.referencenet(reference_latents, t)
                self.reference_control_reader.update(self.referencenet, dtype=dtype)

                # predict the noise residual
                noise_pred = self.unet(
                    sample=latent_model_input,
                    timestep=t,
                    encoder_hidden_states=None,
                    controlnet_cond=control_image,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # # compute predicted original sample (x_0) from sigma-scaled predicted noise
                # pred_original_sample = self.scheduler.get_pred_original_sample(
                #     noise_pred, t, torch.cat([latents] * 2) if do_classifier_free_guidance else latents)
                #
                # # perform guidance
                # if do_classifier_free_guidance:
                #     noise_pred_uncond, noise_pred_text = pred_original_sample.chunk(2)
                #     pred_original_sample = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                #
                # # compute the previous noisy sample x_t -> x_t-1
                # latents = self.scheduler.webui_step(pred_original_sample, t, latents, **extra_step_kwargs)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # Post-processing
        images = self.decode_images(latents)

        # remove cache variables
        if getattr(self.scheduler, 'pred_ori_sample_prev', None) is not None:
            delattr(self.scheduler, 'pred_ori_sample_prev')

        # Offload all models
        self.maybe_free_model_hooks()

        return images
