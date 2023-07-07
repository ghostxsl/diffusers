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

# This model implementation is heavily inspired by https://github.com/haofanwang/ControlNet-for-Diffusers/

import inspect
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, MutableSequence

import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor, CLIPVisionModelWithProjection

from ...image_processor import VaeImageProcessor
from ...loaders import LoraLoaderMixin, TextualInversionLoaderMixin, IPAdapterMixin
from ...models import AutoencoderKL, ControlNetModel, MultiControlNetModel, UNet2DConditionModel
from ...schedulers import KarrasDiffusionSchedulers
from ...utils import (
    is_accelerate_available,
    is_accelerate_version,
    logging,
    replace_example_docstring,
    get_promt_embedding,
)
from ...utils.torch_utils import is_compiled_module, randn_tensor
from ..pipeline_utils import DiffusionPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> # !pip install transformers accelerate
        >>> from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, DDIMScheduler
        >>> from diffusers.utils import load_image
        >>> import numpy as np
        >>> import torch

        >>> init_image = load_image(
        ...     "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_inpaint/boy.png"
        ... )
        >>> init_image = init_image.resize((512, 512))

        >>> generator = torch.Generator(device="cpu").manual_seed(1)

        >>> mask_image = load_image(
        ...     "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_inpaint/boy_mask.png"
        ... )
        >>> mask_image = mask_image.resize((512, 512))


        >>> def make_inpaint_condition(image, image_mask):
        ...     image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
        ...     image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

        ...     assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
        ...     image[image_mask > 0.5] = -1.0  # set as masked pixel
        ...     image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
        ...     image = torch.from_numpy(image)
        ...     return image


        >>> control_image = make_inpaint_condition(init_image, mask_image)

        >>> controlnet = ControlNetModel.from_pretrained(
        ...     "lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float16
        ... )
        >>> pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        ...     "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
        ... )

        >>> pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        >>> pipe.enable_model_cpu_offload()

        >>> # generate image
        >>> image = pipe(
        ...     "a handsome man with ray-ban sunglasses",
        ...     num_inference_steps=20,
        ...     generator=generator,
        ...     eta=1.0,
        ...     image=init_image,
        ...     mask_image=mask_image,
        ...     control_image=control_image,
        ... ).images[0]
        ```
"""


def prepare_mask_and_masked_image(image, mask, height, width, **kwargs):
    assert isinstance(image, Image.Image)
    assert isinstance(mask, Image.Image)

    # preprocess mask
    mask = mask.convert("L").resize((width, height), resample=Image.LANCZOS)

    mask_overlay = np.array(mask, dtype=np.float32) * 2
    mask_overlay = Image.fromarray(np.clip(mask_overlay, 0, 255).astype(np.uint8))

    # preprocess image
    image = image.resize((width, height), resample=Image.LANCZOS)
    image_overlay = Image.new('RGBa', (width, height))
    image_overlay.paste(image.convert("RGBA").convert("RGBa"),
                       mask=ImageOps.invert(mask_overlay))
    image_overlay = image_overlay.convert("RGBA")

    image = np.array(image)[None,].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(torch.float32) / 255.0
    image = 2. * image - 1.

    torch_mask = np.array(mask, dtype=np.float32)[None, None,]
    torch_mask = torch.from_numpy(torch_mask)
    masked_image = image * (torch_mask < 127.5)

    return mask, masked_image, image, image_overlay


class VIPStableDiffusionControlNetInpaintPipeline(DiffusionPipeline, TextualInversionLoaderMixin, LoraLoaderMixin, IPAdapterMixin):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion with ControlNet guidance.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    In addition the pipeline inherits the following loading methods:
        - *Textual-Inversion*: [`loaders.TextualInversionLoaderMixin.load_textual_inversion`]

    <Tip>

    This pipeline can be used both with checkpoints that have been specifically fine-tuned for inpainting, such as
    [runwayml/stable-diffusion-inpainting](https://huggingface.co/runwayml/stable-diffusion-inpainting)
     as well as default text-to-image stable diffusion checkpoints, such as
     [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5).
    Default text-to-image stable diffusion checkpoints might be preferable for controlnets that have been fine-tuned on
    those, such as [lllyasviel/control_v11p_sd15_inpaint](https://huggingface.co/lllyasviel/control_v11p_sd15_inpaint).

    </Tip>

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        controlnet ([`ControlNetModel`] or `List[ControlNetModel]`):
            Provides additional conditioning to the unet during the denoising process. If you set multiple ControlNets
            as a list, the outputs from each ControlNet are added together to create one combined additional
            conditioning.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        feature_extractor ([`CLIPImageProcessor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """
    _optional_components = ["feature_extractor", "image_encoder"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        controlnet: Union[ControlNetModel, List[ControlNetModel], Tuple[ControlNetModel], MultiControlNetModel],
        scheduler: KarrasDiffusionSchedulers,
        feature_extractor: CLIPImageProcessor,
        image_encoder: Optional[CLIPVisionModelWithProjection] = None,
    ):
        super().__init__()

        if isinstance(controlnet, (list, tuple)):
            controlnet = MultiControlNetModel(controlnet)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            controlnet=controlnet,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.control_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False
        )

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_vae_slicing
    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding.

        When this option is enabled, the VAE will split the input tensor in slices to compute decoding in several
        steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.disable_vae_slicing
    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously invoked, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_vae_tiling
    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding.

        When this option is enabled, the VAE will split the input tensor into tiles to compute decoding and encoding in
        several steps. This is useful to save a large amount of memory and to allow the processing of larger images.
        """
        self.vae.enable_tiling()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.disable_vae_tiling
    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously invoked, this method will go back to
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

    def enable_sequential_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, significantly reducing memory usage. When called, unet,
        text_encoder, vae, controlnet, and safety checker have their state dicts saved to CPU and then are moved to a
        `torch.device('meta') and loaded to GPU only when their specific submodule has its `forward` method called.
        Note that offloading happens on a submodule basis. Memory savings are higher than with
        `enable_model_cpu_offload`, but performance is lower.
        """
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae, self.controlnet]:
            cpu_offload(cpu_offloaded_model, device)

    def enable_model_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, reducing memory usage with a low impact on performance. Compared
        to `enable_sequential_cpu_offload`, this method moves one whole model at a time to the GPU when its `forward`
        method is called, and the model remains in GPU until the next model runs. Memory savings are lower than with
        `enable_sequential_cpu_offload`, but performance is much better due to the iterative execution of the `unet`.
        """
        if is_accelerate_available() and is_accelerate_version(">=", "0.17.0.dev0"):
            from accelerate import cpu_offload_with_hook
        else:
            raise ImportError("`enable_model_cpu_offload` requires `accelerate v0.17.0` or higher.")

        device = torch.device(f"cuda:{gpu_id}")

        hook = None
        for cpu_offloaded_model in [self.text_encoder, self.unet, self.vae]:
            _, hook = cpu_offload_with_hook(cpu_offloaded_model, device, prev_module_hook=hook)

        # control net hook has be manually offloaded as it alternates with unet
        cpu_offload_with_hook(self.controlnet, device)

        # We'll offload the last model manually.
        self.final_offload_hook = hook

    @property
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._execution_device
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        `pipeline.enable_sequential_cpu_offload()` the execution device can only be inferred from Accelerate's module
        hooks.
        """
        if not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def _encode_prompt(
        self,
        prompt,
        negative_prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        padding_prompt: bool = False,
        lora_scale: Optional[float] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str`):
                prompt to be encoded.
            negative_prompt (`str`):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance
                 (i.e., ignored if `guidance_scale` is less than `1`).
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        """
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale

        if prompt_embeds is None:
            prompt_embeds = get_promt_embedding(prompt, self.tokenizer, self.text_encoder, device)

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)
        pos_len = prompt_embeds.shape[1]
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(num_images_per_prompt, pos_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            if negative_prompt_embeds is None:
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

    def encode_image(
            self,
            image,
            num_images_per_prompt,
            device,
            dtype,
            do_classifier_free_guidance=False
    ):
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

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents
    def decode_latents(self, latents):
        warnings.warn(
            "The decode_latents method is deprecated and will be removed in a future version. Please"
            " use VaeImageProcessor instead",
            FutureWarning,
        )
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

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
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]

        return timesteps, num_inference_steps - t_start

    def check_inputs(
        self,
        prompt,
        negative_prompt,
        control_image,
        height,
        width,
        callback_steps,
        controlnet_conditioning_scale=1.0,
        control_guidance_start=0.0,
        control_guidance_end=1.0,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and (not isinstance(negative_prompt, str) and not isinstance(negative_prompt, list)):
            raise ValueError(f"`negative_prompt` has to be of type `str` or `list` but is {type(negative_prompt)}")

        # `prompt` needs more sophisticated handling when there are multiple
        # conditionings.
        if isinstance(self.controlnet, MultiControlNetModel):
            if isinstance(prompt, list):
                logger.warning(
                    f"You have {len(self.controlnet.nets)} ControlNets and you have passed {len(prompt)}"
                    " prompts. The conditionings will be fixed across the prompts."
                )

        # Check `image`
        is_compiled = hasattr(F, "scaled_dot_product_attention") and isinstance(
            self.controlnet, torch._dynamo.eval_frame.OptimizedModule
        )
        if (
            isinstance(self.controlnet, ControlNetModel)
            or is_compiled
            and isinstance(self.controlnet._orig_mod, ControlNetModel)
        ):
            self.check_image(control_image, prompt)
        elif (
            isinstance(self.controlnet, MultiControlNetModel)
            or is_compiled
            and isinstance(self.controlnet._orig_mod, MultiControlNetModel)
        ):
            if not isinstance(control_image, list):
                raise TypeError("For multiple controlnets: `image` must be type `list`")

            # When `image` is a nested list:
            # (e.g. [[canny_image_1, pose_image_1], [canny_image_2, pose_image_2]])
            elif any(isinstance(i, list) for i in control_image):
                raise ValueError("A single batch of multiple conditionings are supported at the moment.")
            elif len(control_image) != len(self.controlnet.nets):
                raise ValueError(
                    f"For multiple controlnets: `image` must have the same length as the number of controlnets, "
                    f"but got {len(control_image)} images and {len(self.controlnet.nets)} ControlNets."
                )

            for image_ in control_image:
                self.check_image(image_, prompt)
        else:
            assert False

        # Check `controlnet_conditioning_scale`
        if (
            isinstance(self.controlnet, ControlNetModel)
            or is_compiled
            and isinstance(self.controlnet._orig_mod, ControlNetModel)
        ):
            if not isinstance(controlnet_conditioning_scale, float):
                raise TypeError("For single controlnet: `controlnet_conditioning_scale` must be type `float`.")
        elif (
            isinstance(self.controlnet, MultiControlNetModel)
            or is_compiled
            and isinstance(self.controlnet._orig_mod, MultiControlNetModel)
        ):
            if isinstance(controlnet_conditioning_scale, list):
                if any(isinstance(i, list) for i in controlnet_conditioning_scale):
                    raise ValueError("A single batch of multiple conditionings are supported at the moment.")
            elif isinstance(controlnet_conditioning_scale, list) and len(controlnet_conditioning_scale) != len(
                self.controlnet.nets
            ):
                raise ValueError(
                    "For multiple controlnets: When `controlnet_conditioning_scale` is specified as `list`, it must have"
                    " the same length as the number of controlnets"
                )
        else:
            assert False

        if len(control_guidance_start) != len(control_guidance_end):
            raise ValueError(
                f"`control_guidance_start` has {len(control_guidance_start)} elements, but `control_guidance_end` has {len(control_guidance_end)} elements. Make sure to provide the same number of elements to each list."
            )

        if isinstance(self.controlnet, MultiControlNetModel):
            if len(control_guidance_start) != len(self.controlnet.nets):
                raise ValueError(
                    f"`control_guidance_start`: {control_guidance_start} has {len(control_guidance_start)} elements but there are {len(self.controlnet.nets)} controlnets available. Make sure to provide {len(self.controlnet.nets)}."
                )

        for start, end in zip(control_guidance_start, control_guidance_end):
            if start >= end:
                raise ValueError(
                    f"control guidance start: {start} cannot be larger or equal to control guidance end: {end}."
                )
            if start < 0.0:
                raise ValueError(f"control guidance start: {start} can't be smaller than 0.")
            if end > 1.0:
                raise ValueError(f"control guidance end: {end} can't be larger than 1.0.")

    def check_image(self, image, prompt):
        image_is_pil = isinstance(image, Image.Image)
        image_is_pil_list = isinstance(image, list) and isinstance(image[0], Image.Image)

        if (
            not image_is_pil
            and not image_is_pil_list
        ):
            raise TypeError(
                f"image must be passed and be one of `PIL image` or `list of PIL images`, but is {type(image)}"
            )

        if image_is_pil:
            image_batch_size = 1
        else:
            image_batch_size = len(image)

        if isinstance(prompt, str):
            prompt_batch_size = 1
        else:
            prompt_batch_size = len(prompt)

        if image_batch_size != 1 and image_batch_size != prompt_batch_size:
            raise ValueError(
                f"If image batch size is not 1, image batch size must be same as prompt batch size. image batch size: {image_batch_size}, prompt batch size: {prompt_batch_size}"
            )

    # Copied from diffusers.pipelines.controlnet.pipeline_controlnet.StableDiffusionControlNetPipeline.prepare_image
    def prepare_control_image(
        self,
        image,
        width,
        height,
        num_images_per_prompt,
        device,
        dtype,
    ):
        image = self.control_image_processor.preprocess(
            image, height=height, width=width).to(dtype=torch.float32)
        image = image.repeat_interleave(num_images_per_prompt, dim=0)
        image = image.to(device=device, dtype=dtype)

        return image

    def prepare_latents(
        self,
        image,
        mask,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        timestep=None,
        is_strength_max=True,
        **kwargs
    ):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if (image is None or timestep is None) and not is_strength_max:
            raise ValueError(
                "Since strength < 1. initial latents are to be initialised as a combination of Image + Noise."
                "However, either the image or the noise timestep has not been provided."
            )

        image = image.to(device=device, dtype=dtype)
        image_latents = self._encode_vae_image(image=image, generator=generator)
        if kwargs.get('masked_content', 'original') == 'noise':
            mask = mask[:1]
            image_latents *= (1 - mask)

        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        # if strength is 1. then initialise the latents to noise, else initial to image + noise
        latents = noise if is_strength_max else self.scheduler.add_noise(image_latents, noise, timestep)
        # if pure noise then scale the initial latents by the  Scheduler's init sigma
        latents = latents * self.scheduler.init_noise_sigma if is_strength_max else latents

        return latents, noise, image_latents

    def _default_height_width(self, height, width, image):
        # NOTE: It is possible that a list of images have different
        # dimensions for each image, so just checking the first image
        # is not _exactly_ correct, but it is simple.
        while isinstance(image, list):
            image = image[0]

        if height is None:
            height = (image.height // 8) * 8  # round down to nearest multiple of 8

        if width is None:
            width = (image.width // 8) * 8  # round down to nearest multiple of 8

        return height, width

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint.StableDiffusionInpaintPipeline.prepare_mask_latents
    def prepare_mask_latents(
        self, mask, masked_image, batch_size, height, width, dtype, device, generator
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        # mask = torch.nn.functional.interpolate(mask,
        #     size=(height // self.vae_scale_factor, width // self.vae_scale_factor),
        #     mode='nearest')
        mask = mask.resize((width // self.vae_scale_factor, height // self.vae_scale_factor))
        mask = torch.from_numpy(np.array(mask, dtype=np.float32)[None, None,])
        mask = torch.round(mask / 255.0).to(device=device, dtype=dtype)

        masked_image = masked_image.to(device=device, dtype=dtype)
        masked_image_latents = self._encode_vae_image(masked_image, generator=generator)

        # duplicate mask and masked_image_latents for each generation per prompt, using mps friendly method
        if mask.shape[0] < batch_size:
            if not batch_size % mask.shape[0] == 0:
                raise ValueError(
                    "The passed mask and the required batch size don't match. Masks are supposed to be duplicated to"
                    f" a total batch size of {batch_size}, but {mask.shape[0]} masks were passed. Make sure the number"
                    " of masks that you pass is divisible by the total requested batch size."
                )
            mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)
        if masked_image_latents.shape[0] < batch_size:
            if not batch_size % masked_image_latents.shape[0] == 0:
                raise ValueError(
                    "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                    f" to a total batch size of {batch_size}, but {masked_image_latents.shape[0]} images were passed."
                    " Make sure the number of images that you pass is divisible by the total requested batch size."
                )
            masked_image_latents = masked_image_latents.repeat(batch_size // masked_image_latents.shape[0], 1, 1, 1)

        # aligning device to prevent device errors when concating it with the latent model input
        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
        return mask, masked_image_latents

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

    def _check_enhance_params(self, param, nums):
        if not isinstance(param, (list, tuple, MutableSequence)):
            param = [param,] * nums
        else:
            assert len(param) == nums
        return param

    def _get_enhance_params(self,
                            image,
                            num_inference_steps,
                            enhance,
                            generator,
                            **kwargs):
        # reference: https://github.com/python-pillow/Pillow/blob/main/src/PIL/ImageEnhance.py
        if isinstance(enhance, str):
            assert enhance in ['color', 'contrast', 'brightness', 'sharpness'], (
                "The enhance type can only be one of color, contrast, brightness and sharpness.")
            enhance = [enhance]
        elif isinstance(enhance, (list, tuple, MutableSequence)):
            for e_type in enhance:
                assert e_type in ['color', 'contrast', 'brightness', 'sharpness'], (
                    "The enhance type can only be one of color, contrast, brightness and sharpness.")
        else:
            raise Exception(f"Error enhance type: {type(enhance)}.")

        grey_latents = []
        for e_type in enhance:
            if e_type == 'color':
                image = image * torch.tensor([0.299, 0.587, 0.114])[:, None, None].to(image)
                image = image.sum(1, keepdim=True).tile([1, 3, 1, 1])
                grey_latents.append(self._encode_vae_image(image, generator))
            elif e_type == 'contrast':
                grey_latents.append(self._encode_vae_image(
                    torch.full_like(image, image.mean()), generator))
            elif e_type == 'brightness':
                grey_latents.append(self._encode_vae_image(
                    torch.zeros_like(image), generator))
            elif e_type == 'sharpness':
                w = torch.tensor([1, 1, 1, 1, 5, 1, 1, 1, 1]).reshape([3, 3]) / 13.
                w = w[None, None].tile([3, 1, 1, 1]).to(image)
                image = F.conv2d(image, w, padding='same', groups=3)
                grey_latents.append(self._encode_vae_image(image, generator))

        enhance_scale = self._check_enhance_params(
            kwargs.get('enhance_scale', 0.025), len(enhance))
        exponent = self._check_enhance_params(
            kwargs.get('enhance_exponent', 0), len(enhance))
        reverse = self._check_enhance_params(
            kwargs.get('enhance_reverse', False), len(enhance))
        start_step = self._check_enhance_params(
            kwargs.get('enhance_guidance_start', 0.01), len(enhance))
        start_step = [int(a * num_inference_steps) for a in start_step]
        end_step = self._check_enhance_params(
            kwargs.get('enhance_guidance_end', 0.5), len(enhance))
        end_step = [int(a * num_inference_steps) for a in end_step]

        out_scale = []
        for s, e, rev, exp_, en_s in zip(start_step, end_step, reverse, exponent, enhance_scale):
            single_scale = []
            for i in range(num_inference_steps):
                if i < s or i >= e:
                    single_scale.append(-1.0)
                    continue
                scale = (i - s) / (e - s)
                if rev:
                    scale = 1 - scale
                single_scale.append(scale ** exp_ * en_s + 1)
            out_scale.append(single_scale)
        return grey_latents, out_scale

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: str = "",
        negative_prompt: str = "",
        image: Image.Image = None,
        mask_image: Image.Image = None,
        control_image: Union[
            Image.Image,
            List[Image.Image],
        ] = None,
        ip_adapter_image: Optional[Image.Image] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        strength: float = 0.75,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        padding_prompt: bool = False,
        class_labels: int = None,
        **kwargs
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str`):
                The prompt or prompts to guide the image generation.
            negative_prompt (`str`):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance
                 (i.e., ignored if `guidance_scale` is less than `1`).
            image (`PIL.Image.Image`, `List[PIL.Image.Image]`):
                The ControlNet input condition. ControlNet uses this input condition to generate guidance to Unet. If
                the type is specified as `Torch.FloatTensor`, it is passed to ControlNet as is. `PIL.Image.Image` can
                also be accepted as an image. The dimensions of the output image defaults to `image`'s dimensions. If
                height and/or width are passed, `image` is resized according to them. If multiple ControlNets are
                specified in init, images must be passed as a list such that each element of the list can be correctly
                batched for input to a single controlnet.
            mask_image (`PIL.Image.Image`, `List[PIL.Image.Image]`):
                The inpaint mask, 1 means the area that needs to be generated.
            control_image (`PIL.Image.Image`, `List[PIL.Image.Image]`):
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            strength (`float`, *optional*, defaults to 1.):
                Conceptually, indicates how much to transform the masked portion of the reference `image`. Must be
                between 0 and 1. `image` will be used as a starting point, adding more noise to it the larger the
                `strength`. The number of denoising steps depends on the amount of noise initially added. When
                `strength` is 1, added noise will be maximum and the denoising process will run for the full number of
                iterations specified in `num_inference_steps`. A value of 1, therefore, essentially ignores the masked
                portion of the reference `image`.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
            controlnet_conditioning_scale (`float` or `List[float]`, *optional*, defaults to 0.5):
                The outputs of the controlnet are multiplied by `controlnet_conditioning_scale` before they are added
                to the residual in the original unet. If multiple ControlNets are specified in init, you can set the
                corresponding scale as a list. Note that by default, we use a smaller conditioning scale for inpainting
                than for [`~StableDiffusionControlNetPipeline.__call__`].
            control_guidance_start (`float` or `List[float]`, *optional*, defaults to 0.0):
                The percentage of total steps at which the controlnet starts applying.
            control_guidance_end (`float` or `List[float]`, *optional*, defaults to 1.0):
                The percentage of total steps at which the controlnet stops applying.

        Examples:

        Returns:
            image: a list with the generated images.
        """
        assert isinstance(prompt, str)
        assert isinstance(negative_prompt, str)
        controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet

        # 0. Default height and width to unet
        height, width = self._default_height_width(height, width, image)

        # align format for control guidance
        if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
            control_guidance_start = len(control_guidance_end) * [control_guidance_start]
        elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
            control_guidance_end = len(control_guidance_start) * [control_guidance_end]
        elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
            mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1
            control_guidance_start, control_guidance_end = mult * [control_guidance_start], mult * [
                control_guidance_end
            ]

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            negative_prompt,
            control_image,
            height,
            width,
            callback_steps,
            controlnet_conditioning_scale,
            control_guidance_start,
            control_guidance_end,
        )

        # 2. Define call parameters
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        if isinstance(controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
            controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(controlnet.nets)

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        negative_prompt_embeds, prompt_embeds = self._encode_prompt(
            prompt,
            negative_prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            padding_prompt=padding_prompt,
            lora_scale=text_encoder_lora_scale,
        )
        prompt_embeds_dtype = prompt_embeds.dtype

        # 4. Prepare control image
        if isinstance(controlnet, ControlNetModel):
            control_image = self.prepare_control_image(
                image=control_image,
                width=width,
                height=height,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=controlnet.dtype,
            ) if controlnet_conditioning_scale != 0 else None
        elif isinstance(controlnet, MultiControlNetModel):
            control_images = []
            for control_image_, control_scale_ in zip(control_image, controlnet_conditioning_scale):
                control_image_ = self.prepare_control_image(
                    image=control_image_,
                    width=width,
                    height=height,
                    num_images_per_prompt=num_images_per_prompt,
                    device=device,
                    dtype=controlnet.dtype,
                ) if control_scale_ != 0 else None
                control_images.append(control_image_)

            control_image = control_images
        else:
            assert False

        # 4.1 Preprocess mask and image - resizes image and mask w.r.t height and width
        mask, masked_image, init_image, image_overlay = prepare_mask_and_masked_image(
            image, mask_image, height, width, **kwargs
        )

        # 5. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = self.get_timesteps(
            num_inference_steps=num_inference_steps, strength=strength
        )
        # at which timestep to set the initial noise (n.b. 50% if strength is 0.5)
        latent_timestep = timesteps[:1].repeat(num_images_per_prompt)
        # create a boolean to check if the strength is set to 1. if so then initialise the latents with pure noise
        is_strength_max = strength == 1.0

        # 6. Prepare mask latent variables
        mask, masked_image_latents = self.prepare_mask_latents(
            mask,
            masked_image,
            num_images_per_prompt,
            height,
            width,
            prompt_embeds_dtype,
            device,
            generator,
        )

        # 6.1 Add image embeds for IP-Adapter
        if self.image_encoder is not None:
            image_embeds, uncond_image_embeds = self.encode_image(
                ip_adapter_image,
                num_images_per_prompt,
                device,
                prompt_embeds_dtype,
                do_classifier_free_guidance,
            )
            if ip_adapter_image is None:
                image_embeds = torch.zeros_like(image_embeds)
                uncond_image_embeds = torch.zeros_like(uncond_image_embeds)

        # 6.2 Add class embeddings
        if self.unet.class_embedding is not None and class_labels is not None:
            class_labels = torch.LongTensor([class_labels])
            class_labels = class_labels.repeat_interleave(num_images_per_prompt, dim=0)
            class_labels = class_labels.to(device=device)

        # 7. Prepare latent variables
        num_channels_latents = self.vae.config.latent_channels
        num_channels_unet = self.unet.config.in_channels
        latents, noise, image_latents = self.prepare_latents(
            init_image,
            mask,
            num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds_dtype,
            device,
            generator,
            timestep=latent_timestep,
            is_strength_max=is_strength_max,
            **kwargs
        )

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7.1 Create tensor stating which controlnets to keep
        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1 - int(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(keeps[0] if len(keeps) == 1 else keeps)

        # 7.2 Prepare image enhance parameters
        enhance = kwargs.pop('enhance', None)
        if enhance is not None:
            grey_latents, enhance_scales = self._get_enhance_params(
                init_image.to(device=device, dtype=prompt_embeds_dtype),
                num_inference_steps, enhance, generator, **kwargs)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if enhance is not None:
                    for grey_latent, enhance_scale in zip(grey_latents, enhance_scales):
                        if enhance_scale[i] >= 0:
                            latents = torch.lerp(grey_latent, latents, enhance_scale[i])

                # scale the latents
                latent_model_input = self.scheduler.scale_model_input(latents, t)

                # controlnet(s) inference
                if isinstance(controlnet_keep[i], list):
                    cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
                else:
                    cond_scale = controlnet_conditioning_scale * controlnet_keep[i]

                if not do_classifier_free_guidance:
                    down_block_res_samples, mid_block_res_sample = self.controlnet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        controlnet_cond=control_image,
                        conditioning_scale=cond_scale,
                        return_dict=False,
                    )
                else:
                    # forward twice
                    down_neg, mid_neg = self.controlnet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=negative_prompt_embeds,
                        controlnet_cond=control_image,
                        conditioning_scale=cond_scale,
                        return_dict=False,
                    )
                    down_pos, mid_pos = self.controlnet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        controlnet_cond=control_image,
                        conditioning_scale=cond_scale,
                        return_dict=False,
                    )
                    down_block_res_samples = [down_neg, down_pos]
                    mid_block_res_sample = [mid_neg, mid_pos]

                # predict the noise residual
                if num_channels_unet == 9:
                    latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)

                if not do_classifier_free_guidance:
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                        added_cond_kwargs={'image_embeds': [image_embeds.unsqueeze(1)]} if self.image_encoder is not None else None,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                        return_dict=False,
                    )[0]
                else:
                    # forward twice
                    noise_pred_neg = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=negative_prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                        added_cond_kwargs={'image_embeds': [uncond_image_embeds.unsqueeze(1)]} if self.image_encoder is not None else None,
                        down_block_additional_residuals=down_block_res_samples[0],
                        mid_block_additional_residual=mid_block_res_sample[0],
                        class_labels=class_labels if self.unet.class_embedding is not None else None,
                        return_dict=False,
                    )[0]
                    noise_pred_pos = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                        added_cond_kwargs={'image_embeds': [image_embeds.unsqueeze(1)]} if self.image_encoder is not None else None,
                        down_block_additional_residuals=down_block_res_samples[1],
                        mid_block_additional_residual=mid_block_res_sample[1],
                        class_labels=class_labels if self.unet.class_embedding is not None else None,
                        return_dict=False,
                    )[0]
                    noise_pred = torch.cat([noise_pred_neg, noise_pred_pos])

                # compute predicted original sample (x_0) from sigma-scaled predicted noise
                pred_original_sample = self.scheduler.get_pred_original_sample(
                    noise_pred, t, torch.cat([latents] * 2) if do_classifier_free_guidance else latents)

                # perform guidance
                if do_classifier_free_guidance:
                    pred_uncond, pred_text = pred_original_sample.chunk(2)
                    pred_original_sample = pred_uncond + guidance_scale * (pred_text - pred_uncond)

                if num_channels_unet == 4:
                    # inpaint
                    init_latents_proper = image_latents[:1]
                    init_mask = mask[:1]
                    pred_original_sample = (1 - init_mask) * init_latents_proper + init_mask * pred_original_sample

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.webui_step(pred_original_sample, t, latents, **extra_step_kwargs)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        if num_channels_unet == 4:
            # inpaint
            init_mask = mask[:1]
            latents = (1 - init_mask) * image_latents[:1] + init_mask * latents
        # If we do sequential model offloading, let's offload unet and controlnet
        # manually for max memory savings
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.unet.to("cpu")
            self.controlnet.to("cpu")
            torch.cuda.empty_cache()

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        else:
            image = latents

        do_denormalize = [True] * image.shape[0]
        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # remove cache variables
        if getattr(self.scheduler, 'pred_ori_sample_prev', None) is not None:
            delattr(self.scheduler, 'pred_ori_sample_prev')

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        return image, image_overlay
