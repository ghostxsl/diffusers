# Copyright 2023 The HuggingFace Team. All rights reserved.
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
# limitations under the License.
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.normalization import GroupNorm

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention_processor import USE_PEFT_BACKEND, AttentionProcessor
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.models.lora import LoRACompatibleConv
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.unets.unet_2d_blocks import (
    UNetMidBlock2DCrossAttn,
    ResnetBlock2D,
    Transformer2DModel,
    Downsample2D,
    Upsample2D,
)
from diffusers.models.unets.unet_3d_blocks import (
    CrossAttnDownBlockMotion,
    DownBlockMotion,
    UNetMidBlockCrossAttnMotion,
    TransformerTemporalModel,
    get_down_block,
)
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.controlnetxs import ControlNetXSModel
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.utils import BaseOutput, logging


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class ControlNetXSOutput(BaseOutput):
    """
    The output of [`ControlNetXSModel`].

    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            The output of the `ControlNetXSModel`. Unlike `ControlNetOutput` this is NOT to be added to the base model
            output, but is already the final output.
    """

    sample: torch.FloatTensor = None


# copied from diffusers.models.controlnet.ControlNetConditioningEmbedding
class ControlNetConditioningEmbedding(nn.Module):
    """
    Quoting from https://arxiv.org/abs/2302.05543: "Stable Diffusion uses a pre-processing method similar to VQ-GAN
    [11] to convert the entire dataset of 512 × 512 images into smaller 64 × 64 “latent images” for stabilized
    training. This requires ControlNets to convert image-based conditions to 64 × 64 feature space to match the
    convolution size. We use a tiny network E(·) of four convolution layers with 4 × 4 kernels and 2 × 2 strides
    (activated by ReLU, channels are 16, 32, 64, 128, initialized with Gaussian weights, trained jointly with the full
    model) to encode image-space conditions ... into feature maps ..."
    """

    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (16, 32, 96, 256),
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(conditioning_channels, block_out_channels[0], kernel_size=3, padding=1)

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))

        self.conv_out = zero_module(
            nn.Conv2d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1)
        )

    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        embedding = self.conv_out(embedding)

        return embedding


class ControlNetXSMotionModel(ModelMixin, ConfigMixin):
    r"""
    A ControlNet-XS model

    This model inherits from [`ModelMixin`] and [`ConfigMixin`]. Check the superclass documentation for it's generic
    methods implemented for all models (such as downloading or saving).

    Most of parameters for this model are passed into the [`UNet2DConditionModel`] it creates. Check the documentation
    of [`UNet2DConditionModel`] for them.

    Parameters:
        conditioning_channels (`int`, defaults to 3):
            Number of channels of conditioning input (e.g. an image)
        controlnet_conditioning_channel_order (`str`, defaults to `"rgb"`):
            The channel order of conditional image. Will convert to `rgb` if it's `bgr`.
        conditioning_embedding_out_channels (`tuple[int]`, defaults to `(16, 32, 96, 256)`):
            The tuple of output channel for each block in the `controlnet_cond_embedding` layer.
        time_embedding_input_dim (`int`, defaults to 320):
            Dimension of input into time embedding. Needs to be same as in the base model.
        time_embedding_dim (`int`, defaults to 1280):
            Dimension of output from time embedding. Needs to be same as in the base model.
        learn_embedding (`bool`, defaults to `False`):
            Whether to use time embedding of the control model. If yes, the time embedding is a linear interpolation of
            the time embeddings of the control and base model with interpolation parameter `time_embedding_mix**3`.
        time_embedding_mix (`float`, defaults to 1.0):
            Linear interpolation parameter used if `learn_embedding` is `True`. A value of 1.0 means only the
            control model's time embedding will be used. A value of 0.0 means only the base model's time embedding will be used.
        base_model_channel_sizes (`Dict[str, List[Tuple[int]]]`):
            Channel sizes of each subblock of base model. Use `gather_subblock_sizes` on your base model to compute it.
    """

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 4,
        conditioning_channels: int = 3,
        conditioning_embedding_out_channels: Tuple[int] = (16, 32, 96, 256),
        controlnet_conditioning_channel_order: str = "rgb",
        time_embedding_input_dim: int = 320,
        time_embedding_dim: int = 1280,
        time_embedding_mix: float = 1.0,
        learn_embedding: bool = False,
        base_model_channel_sizes: Dict[str, List[Tuple[int]]] = {
            "down": [
                (4, 320),
                (320, 320),
                (320, 320),
                (320, 320),
                (320, 640),
                (640, 640),
                (640, 640),
                (640, 1280),
                (1280, 1280),
                (1280, 1280),
                (1280, 1280),
                (1280, 1280),
            ],
            "mid": [(1280, 1280)],
            "up": [
                (2560, 1280),
                (2560, 1280),
                (2560, 1280),
                (2560, 1280),
                (2560, 1280),
                (1920, 1280),
                (1920, 640),
                (1280, 640),
                (960, 640),
                (960, 320),
                (640, 320),
                (640, 320),
            ],
        },
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlockMotion",
            "CrossAttnDownBlockMotion",
            "CrossAttnDownBlockMotion",
            "DownBlockMotion",
        ),
        block_out_channels: Tuple[int] = (32, 64, 128, 128),
        layers_per_block: int = 2,
        act_fn: str = "silu",
        norm_num_groups: Optional[int] = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: Union[int, Tuple[int]] = 768,
        use_linear_projection: bool = False,
        num_attention_heads: Optional[Union[int, Tuple[int]]] = (4, 8, 16, 16),
        motion_max_seq_length: int = 32,
        use_motion_mid_block: int = True,
    ):
        super().__init__()

        self.sample_size = sample_size

        # conv input
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=1)

        # time
        if learn_embedding:
            self.time_embedding = TimestepEmbedding(
                time_embedding_input_dim,
                time_embedding_dim,
                act_fn=act_fn,
            )

        self.down_blocks = nn.ModuleList([])

        if isinstance(num_attention_heads, int):
            num_attention_heads = (num_attention_heads,) * len(down_block_types)

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embedding_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                num_attention_heads=num_attention_heads[i],
                downsample_padding=1,
                use_linear_projection=use_linear_projection,
                dual_cross_attention=False,
                temporal_num_attention_heads=num_attention_heads[i],
                temporal_max_seq_length=motion_max_seq_length,
            )
            self.down_blocks.append(down_block)

        # mid
        if use_motion_mid_block:
            self.mid_block = UNetMidBlockCrossAttnMotion(
                in_channels=block_out_channels[-1],
                temb_channels=time_embedding_dim,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                cross_attention_dim=cross_attention_dim,
                num_attention_heads=num_attention_heads[-1],
                resnet_groups=norm_num_groups,
                dual_cross_attention=False,
                use_linear_projection=use_linear_projection,
                temporal_num_attention_heads=num_attention_heads[-1],
                temporal_max_seq_length=motion_max_seq_length,
            )
        else:
            self.mid_block = UNetMidBlock2DCrossAttn(
                in_channels=block_out_channels[-1],
                temb_channels=time_embedding_dim,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                cross_attention_dim=cross_attention_dim,
                num_attention_heads=num_attention_heads[-1],
                resnet_groups=norm_num_groups,
                dual_cross_attention=False,
                use_linear_projection=use_linear_projection,
            )

        # 2 - Do model surgery on control model

        # We concat the output of each base encoder subblocks to the input of the next control encoder subblock
        # (We ignore the 1st element, as it represents the `conv_in`.)
        extra_input_channels = [input_channels for input_channels, _ in base_model_channel_sizes["down"][1:]]
        it_extra_input_channels = iter(extra_input_channels)

        for b, block in enumerate(self.down_blocks):
            for r in range(len(block.resnets)):
                increase_block_input_in_encoder_resnet(
                    self, block_no=b, resnet_idx=r, by=next(it_extra_input_channels)
                )

            if block.downsamplers:
                increase_block_input_in_encoder_downsampler(
                    self, block_no=b, by=next(it_extra_input_channels)
                )

        increase_block_input_in_mid_resnet(self, by=extra_input_channels[-1])

        # 2.3 - Make group norms work with modified channel sizes
        adjust_group_norms(self)

        # 3 - Gather Channel Sizes
        self.ch_inout_ctrl = self._gather_subblock_sizes()
        self.ch_inout_base = base_model_channel_sizes

        # 4 - Build connections between base and control model
        self.down_zero_convs_out = nn.ModuleList([])
        self.down_zero_convs_in = nn.ModuleList([])
        self.middle_block_out = nn.ModuleList([])
        self.up_zero_convs_out = nn.ModuleList([])

        for ch_io_base in self.ch_inout_base["down"]:
            self.down_zero_convs_in.append(
                self._make_zero_conv(in_channels=ch_io_base[1], out_channels=ch_io_base[1])
            )
        for ch_io_ctrl, ch_io_base in zip(self.ch_inout_ctrl["down"], self.ch_inout_base["down"]):
            self.down_zero_convs_out.append(
                self._make_zero_conv(in_channels=ch_io_ctrl[1], out_channels=ch_io_base[1])
            )

        self.middle_block_out = self._make_zero_conv(
            self.ch_inout_ctrl["mid"][-1][1], self.ch_inout_base["mid"][-1][1]
        )

        self.up_zero_convs_out.append(
            self._make_zero_conv(self.ch_inout_ctrl["down"][-1][1], self.ch_inout_base["mid"][-1][1])
        )
        for i in range(1, len(self.ch_inout_ctrl["down"])):
            self.up_zero_convs_out.append(
                self._make_zero_conv(self.ch_inout_ctrl["down"][-(i + 1)][1], self.ch_inout_base["up"][i - 1][1])
            )

        # 5 - Create conditioning hint embedding
        self.controlnet_cond_embedding = ControlNetConditioningEmbedding(
            conditioning_embedding_channels=block_out_channels[0],
            block_out_channels=conditioning_embedding_out_channels,
            conditioning_channels=conditioning_channels,
        )

    @classmethod
    def from_controlnet2d(
        cls,
        controlnet: ControlNetXSModel
    ):
        config = controlnet.config
        config["_class_name"] = cls.__name__

        down_blocks = []
        for down_blocks_type in config["down_block_types"]:
            if "CrossAttn" in down_blocks_type:
                down_blocks.append("CrossAttnDownBlockMotion")
            else:
                down_blocks.append("DownBlockMotion")
        config["down_block_types"] = down_blocks

        model = cls.from_config(config)

        state_dict = controlnet.state_dict()
        state_dict = {k.replace("control_model.", ""): v for k, v in state_dict.items()}
        m, u = model.load_state_dict(state_dict, strict=False)

        return model

    def freeze_controlnet2d_params(self):
        # Freeze everything
        for param in self.parameters():
            param.requires_grad = False

        # Unfreeze Motion Modules
        for down_block in self.down_blocks:
            motion_modules = down_block.motion_modules
            for param in motion_modules.parameters():
                param.requires_grad = True

        if hasattr(self.mid_block, "motion_modules"):
            motion_modules = self.mid_block.motion_modules
            for param in motion_modules.parameters():
                param.requires_grad = True

        for param in self.down_zero_convs_out.parameters():
            param.requires_grad = True
        for param in self.down_zero_convs_in.parameters():
            param.requires_grad = True
        for param in self.middle_block_out.parameters():
            param.requires_grad = True
        for param in self.up_zero_convs_out.parameters():
            param.requires_grad = True

    def _gather_subblock_sizes(self):
        channel_sizes = {"down": [], "mid": [], "up": []}

        # input convolution
        channel_sizes["down"].append((self.conv_in.in_channels, self.conv_in.out_channels))

        # encoder blocks
        for module in self.down_blocks:
            if isinstance(module, (CrossAttnDownBlockMotion, DownBlockMotion)):
                for r in module.resnets:
                    channel_sizes["down"].append((r.in_channels, r.out_channels))
                if module.downsamplers:
                    channel_sizes["down"].append(
                        (module.downsamplers[0].channels, module.downsamplers[0].out_channels)
                    )
            else:
                raise ValueError(f"Encountered unknown module of type {type(module)} while creating ControlNet-XS.")

        # middle block
        channel_sizes["mid"].append((self.mid_block.resnets[0].in_channels, self.mid_block.resnets[0].out_channels))

        return channel_sizes

    def _make_zero_conv(self, in_channels, out_channels):
        return zero_module(nn.Conv2d(in_channels, out_channels, 1, padding=0))

    @property
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        return self.attn_processors

    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        self.set_attn_processor(processor)

    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        self.set_default_attn_processor()

    def set_attention_slice(self, slice_size):
        r"""
        Enable sliced attention computation.

        When this option is enabled, the attention module splits the input tensor in slices to compute attention in
        several steps. This is useful for saving some memory in exchange for a small decrease in speed.

        Args:
            slice_size (`str` or `int` or `list(int)`, *optional*, defaults to `"auto"`):
                When `"auto"`, input to the attention heads is halved, so attention is computed in two steps. If
                `"max"`, maximum amount of memory is saved by running only one slice at a time. If a number is
                provided, uses as many slices as `attention_head_dim // slice_size`. In this case, `attention_head_dim`
                must be a multiple of `slice_size`.
        """
        self.set_attention_slice(slice_size)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (UNet2DConditionModel)):
            if value:
                module.enable_gradient_checkpointing()
            else:
                module.disable_gradient_checkpointing()

    def forward(
        self,
        base_model: UNet2DConditionModel,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        controlnet_cond: torch.Tensor,
        conditioning_scale: float = 1.0,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        return_dict: bool = True,
        **kwargs,
    ) -> Union[ControlNetXSOutput, Tuple]:
        # check channel order
        channel_order = self.config.controlnet_conditioning_channel_order

        if channel_order == "rgb":
            # in rgb order by default
            ...
        elif channel_order == "bgr":
            controlnet_cond = torch.flip(controlnet_cond, dims=[2])
        else:
            raise ValueError(f"unknown `controlnet_conditioning_channel_order`: {channel_order}")

        # scale control strength
        n_connections = len(self.down_zero_convs_out) + 1 + len(self.up_zero_convs_out)
        scale_list = torch.full((n_connections,), conditioning_scale)

        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])
        # (batch, num_frame, channel, height, width)
        num_frames = sample.shape[1]

        t_emb = base_model.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)

        if self.config.learn_embedding:
            ctrl_temb = self.time_embedding(t_emb, timestep_cond)
            base_temb = base_model.time_embedding(t_emb, timestep_cond)
            interpolation_param = self.config.time_embedding_mix**0.3

            temb = ctrl_temb * interpolation_param + base_temb * (1 - interpolation_param)
        else:
            temb = base_model.time_embedding(t_emb, timestep_cond)

        # added time & text embeddings
        aug_emb = None

        if base_model.config.addition_embed_type is not None:
            if base_model.config.addition_embed_type == "text":
                aug_emb = base_model.add_embedding(encoder_hidden_states)
            elif base_model.config.addition_embed_type == "text_image":
                raise NotImplementedError()
            elif base_model.config.addition_embed_type == "text_time":
                # SDXL - style
                if "text_embeds" not in added_cond_kwargs:
                    raise ValueError(
                        f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `text_embeds` to be passed in `added_cond_kwargs`"
                    )
                text_embeds = added_cond_kwargs.get("text_embeds")
                if "time_ids" not in added_cond_kwargs:
                    raise ValueError(
                        f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `time_ids` to be passed in `added_cond_kwargs`"
                    )
                time_ids = added_cond_kwargs.get("time_ids")
                time_embeds = base_model.add_time_proj(time_ids.flatten())
                time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))
                add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
                add_embeds = add_embeds.to(temb.dtype)
                aug_emb = base_model.add_embedding(add_embeds)
            elif base_model.config.addition_embed_type == "image":
                raise NotImplementedError()
            elif base_model.config.addition_embed_type == "image_hint":
                raise NotImplementedError()

        temb = temb + aug_emb if aug_emb is not None else temb
        temb = temb.repeat_interleave(repeats=num_frames, dim=0)

        # text embeddings
        cemb = encoder_hidden_states.repeat_interleave(repeats=num_frames, dim=0)

        # Preparation
        controlnet_cond = controlnet_cond.reshape((-1,) + controlnet_cond.shape[-3:])
        guided_hint = self.controlnet_cond_embedding(controlnet_cond)

        # [b, f, c, h, w]
        sample = sample.reshape((-1,) + sample.shape[-3:])
        h_ctrl = h_base = sample
        hs_base, hs_ctrl = [], []
        it_down_convs_in, it_down_convs_out, it_up_convs_out = map(
            iter, (self.down_zero_convs_in, self.down_zero_convs_out, self.up_zero_convs_out)
        )
        scales = iter(scale_list)

        base_down_subblocks = to_sub_blocks(base_model.down_blocks)
        ctrl_down_subblocks = to_sub_blocks(self.down_blocks)
        base_mid_subblocks = to_sub_blocks([base_model.mid_block])
        ctrl_mid_subblocks = to_sub_blocks([self.mid_block])
        base_up_subblocks = to_sub_blocks(base_model.up_blocks)

        # Cross Control
        # 0 - conv in
        h_base = base_model.conv_in(h_base)
        h_ctrl = self.conv_in(h_ctrl)
        if guided_hint is not None:
            h_ctrl += guided_hint
        h_base = h_base + next(it_down_convs_out)(h_ctrl) * next(scales)  # D - add ctrl -> base

        hs_base.append(h_base)
        hs_ctrl.append(h_ctrl)

        # 1 - down
        for m_base, m_ctrl in zip(base_down_subblocks, ctrl_down_subblocks):
            h_ctrl = torch.cat([h_ctrl, next(it_down_convs_in)(h_base)], dim=1)  # A - concat base -> ctrl
            h_base = m_base(h_base, temb, cemb, num_frames, attention_mask, cross_attention_kwargs)  # B - apply base subblock
            h_ctrl = m_ctrl(h_ctrl, temb, cemb, num_frames, attention_mask, cross_attention_kwargs)  # C - apply ctrl subblock
            h_base = h_base + next(it_down_convs_out)(h_ctrl) * next(scales)  # D - add ctrl -> base
            hs_base.append(h_base)
            hs_ctrl.append(h_ctrl)

        # 2 - mid
        h_ctrl = torch.cat([h_ctrl, next(it_down_convs_in)(h_base)], dim=1)  # A - concat base -> ctrl
        for m_base, m_ctrl in zip(base_mid_subblocks, ctrl_mid_subblocks):
            h_base = m_base(h_base, temb, cemb, num_frames, attention_mask, cross_attention_kwargs)  # B - apply base subblock
            h_ctrl = m_ctrl(h_ctrl, temb, cemb, num_frames, attention_mask, cross_attention_kwargs)  # C - apply ctrl subblock
        h_base = h_base + self.middle_block_out(h_ctrl) * next(scales)  # D - add ctrl -> base

        # 3 - up
        for i, m_base in enumerate(base_up_subblocks):
            h_base = h_base + next(it_up_convs_out)(hs_ctrl.pop()) * next(scales)  # add info from ctrl encoder
            h_base = torch.cat([h_base, hs_base.pop()], dim=1)  # concat info from base encoder+ctrl encoder
            h_base = m_base(h_base, temb, cemb, num_frames, attention_mask, cross_attention_kwargs)

        h_base = base_model.conv_norm_out(h_base)
        h_base = base_model.conv_act(h_base)
        h_base = base_model.conv_out(h_base)

        h_base = h_base.reshape((-1, num_frames) + h_base.shape[-3:])

        if not return_dict:
            return (h_base,)

        return ControlNetXSOutput(sample=h_base)

    @torch.no_grad()
    def _check_if_vae_compatible(self, vae: AutoencoderKL):
        condition_downscale_factor = 2 ** (len(self.config.conditioning_embedding_out_channels) - 1)
        vae_downscale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        compatible = condition_downscale_factor == vae_downscale_factor
        return compatible, condition_downscale_factor, vae_downscale_factor


class SubBlock(nn.ModuleList):
    """A SubBlock is the largest piece of either base or control model, that is executed independently of the other model respectively.
    Before each subblock, information is concatted from base to control. And after each subblock, information is added from control to base.
    """

    def __init__(self, ms, *args, **kwargs):
        if not is_iterable(ms):
            ms = [ms]
        super().__init__(ms, *args, **kwargs)

    def forward(
        self,
        x: torch.Tensor,
        temb: torch.Tensor,
        cemb: torch.Tensor,
        num_frames: int = 1,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Iterate through children and pass correct information to each."""
        for m in self:
            if isinstance(m, ResnetBlock2D):
                x = m(x, temb)
            elif isinstance(m, Transformer2DModel):
                x = m(
                    x,
                    cemb,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]
            elif isinstance(m, TransformerTemporalModel):
                x = m(
                    x,
                    num_frames=num_frames,
                    return_dict=False,
                )[0]
            elif isinstance(m, Downsample2D):
                x = m(x)
            elif isinstance(m, Upsample2D):
                x = m(x)
            else:
                raise ValueError(
                    f"Type of m is {type(m)} but should be `ResnetBlock2D`, `Transformer2DModel`,  `Downsample2D` or `Upsample2D`"
                )

        return x


def increase_block_input_in_encoder_resnet(unet: ControlNetXSMotionModel, block_no, resnet_idx, by):
    """Increase channels sizes to allow for additional concatted information from base model"""
    r = unet.down_blocks[block_no].resnets[resnet_idx]
    old_norm1, old_conv1 = r.norm1, r.conv1
    # norm
    norm_args = "num_groups num_channels eps affine".split(" ")
    for a in norm_args:
        assert hasattr(old_norm1, a)
    norm_kwargs = {a: getattr(old_norm1, a) for a in norm_args}
    norm_kwargs["num_channels"] += by  # surgery done here
    # conv1
    conv1_args = [
        "in_channels",
        "out_channels",
        "kernel_size",
        "stride",
        "padding",
        "dilation",
        "groups",
        "bias",
        "padding_mode",
    ]
    if not USE_PEFT_BACKEND:
        conv1_args.append("lora_layer")

    for a in conv1_args:
        assert hasattr(old_conv1, a)

    conv1_kwargs = {a: getattr(old_conv1, a) for a in conv1_args}
    conv1_kwargs["bias"] = "bias" in conv1_kwargs  # as param, bias is a boolean, but as attr, it's a tensor.
    conv1_kwargs["in_channels"] += by  # surgery done here
    # conv_shortcut
    # as we changed the input size of the block, the input and output sizes are likely different,
    # therefore we need a conv_shortcut (simply adding won't work)
    conv_shortcut_args_kwargs = {
        "in_channels": conv1_kwargs["in_channels"],
        "out_channels": conv1_kwargs["out_channels"],
        # default arguments from resnet.__init__
        "kernel_size": 1,
        "stride": 1,
        "padding": 0,
        "bias": True,
    }
    # swap old with new modules
    unet.down_blocks[block_no].resnets[resnet_idx].norm1 = GroupNorm(**norm_kwargs)
    unet.down_blocks[block_no].resnets[resnet_idx].conv1 = (
        nn.Conv2d(**conv1_kwargs) if USE_PEFT_BACKEND else LoRACompatibleConv(**conv1_kwargs)
    )
    unet.down_blocks[block_no].resnets[resnet_idx].conv_shortcut = (
        nn.Conv2d(**conv_shortcut_args_kwargs) if USE_PEFT_BACKEND else LoRACompatibleConv(**conv_shortcut_args_kwargs)
    )
    unet.down_blocks[block_no].resnets[resnet_idx].in_channels += by  # surgery done here


def increase_block_input_in_encoder_downsampler(unet: ControlNetXSMotionModel, block_no, by):
    """Increase channels sizes to allow for additional concatted information from base model"""
    old_down = unet.down_blocks[block_no].downsamplers[0].conv

    args = [
        "in_channels",
        "out_channels",
        "kernel_size",
        "stride",
        "padding",
        "dilation",
        "groups",
        "bias",
        "padding_mode",
    ]
    if not USE_PEFT_BACKEND:
        args.append("lora_layer")

    for a in args:
        assert hasattr(old_down, a)
    kwargs = {a: getattr(old_down, a) for a in args}
    kwargs["bias"] = "bias" in kwargs  # as param, bias is a boolean, but as attr, it's a tensor.
    kwargs["in_channels"] += by  # surgery done here
    # swap old with new modules
    unet.down_blocks[block_no].downsamplers[0].conv = (
        nn.Conv2d(**kwargs) if USE_PEFT_BACKEND else LoRACompatibleConv(**kwargs)
    )
    unet.down_blocks[block_no].downsamplers[0].channels += by  # surgery done here


def increase_block_input_in_mid_resnet(unet: ControlNetXSMotionModel, by):
    """Increase channels sizes to allow for additional concatted information from base model"""
    m = unet.mid_block.resnets[0]
    old_norm1, old_conv1 = m.norm1, m.conv1
    # norm
    norm_args = "num_groups num_channels eps affine".split(" ")
    for a in norm_args:
        assert hasattr(old_norm1, a)
    norm_kwargs = {a: getattr(old_norm1, a) for a in norm_args}
    norm_kwargs["num_channels"] += by  # surgery done here
    conv1_args = [
        "in_channels",
        "out_channels",
        "kernel_size",
        "stride",
        "padding",
        "dilation",
        "groups",
        "bias",
        "padding_mode",
    ]
    if not USE_PEFT_BACKEND:
        conv1_args.append("lora_layer")

    conv1_kwargs = {a: getattr(old_conv1, a) for a in conv1_args}
    conv1_kwargs["bias"] = "bias" in conv1_kwargs  # as param, bias is a boolean, but as attr, it's a tensor.
    conv1_kwargs["in_channels"] += by  # surgery done here
    # conv_shortcut
    # as we changed the input size of the block, the input and output sizes are likely different,
    # therefore we need a conv_shortcut (simply adding won't work)
    conv_shortcut_args_kwargs = {
        "in_channels": conv1_kwargs["in_channels"],
        "out_channels": conv1_kwargs["out_channels"],
        # default arguments from resnet.__init__
        "kernel_size": 1,
        "stride": 1,
        "padding": 0,
        "bias": True,
    }
    # swap old with new modules
    unet.mid_block.resnets[0].norm1 = GroupNorm(**norm_kwargs)
    unet.mid_block.resnets[0].conv1 = (
        nn.Conv2d(**conv1_kwargs) if USE_PEFT_BACKEND else LoRACompatibleConv(**conv1_kwargs)
    )
    unet.mid_block.resnets[0].conv_shortcut = (
        nn.Conv2d(**conv_shortcut_args_kwargs) if USE_PEFT_BACKEND else LoRACompatibleConv(**conv_shortcut_args_kwargs)
    )
    unet.mid_block.resnets[0].in_channels += by  # surgery done here


def adjust_group_norms(unet: ControlNetXSMotionModel, max_num_group: int = 32):
    def find_denominator(number, start):
        if start >= number:
            return number
        while start != 0:
            residual = number % start
            if residual == 0:
                return start
            start -= 1

    for block in [*unet.down_blocks, unet.mid_block]:
        # resnets
        for r in block.resnets:
            if r.norm1.num_groups < max_num_group:
                r.norm1.num_groups = find_denominator(r.norm1.num_channels, start=max_num_group)

            if r.norm2.num_groups < max_num_group:
                r.norm2.num_groups = find_denominator(r.norm2.num_channels, start=max_num_group)

        # transformers
        if hasattr(block, "attentions"):
            for a in block.attentions:
                if a.norm.num_groups < max_num_group:
                    a.norm.num_groups = find_denominator(a.norm.num_channels, start=max_num_group)


def is_iterable(o):
    if isinstance(o, str):
        return False
    try:
        iter(o)
        return True
    except TypeError:
        return False


def to_sub_blocks(blocks):
    if not is_iterable(blocks):
        blocks = [blocks]

    sub_blocks = []

    for b in blocks:
        if hasattr(b, "resnets"):
            if hasattr(b, "attentions") and b.attentions is not None:
                for r, a, m in zip(b.resnets, b.attentions, b.motion_modules):
                    sub_blocks.append([r, a, m])

                num_resnets = len(b.resnets)
                num_attns = len(b.attentions)

                if num_resnets > num_attns:
                    # we can have more resnets than attentions, so add each resnet as separate subblock
                    for i in range(num_attns, num_resnets):
                        sub_blocks.append([b.resnets[i]])
            else:
                for r, m in zip(b.resnets, b.motion_modules):
                    sub_blocks.append([r, m])

        # upsamplers are part of the same subblock
        if hasattr(b, "upsamplers") and b.upsamplers is not None:
            for u in b.upsamplers:
                sub_blocks[-1].extend([u])

        # downsamplers are own subblock
        if hasattr(b, "downsamplers") and b.downsamplers is not None:
            for d in b.downsamplers:
                sub_blocks.append([d])

    return list(map(SubBlock, sub_blocks))


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module
