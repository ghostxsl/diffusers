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
# limitations under the License.

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn

from diffusers.models.vip.transformer_flux import (
    FluxTransformer2DModel,
    FluxSingleTransformerBlock,
    FluxTransformerBlock,
)
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention_processor import AttentionProcessor
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.controlnets.controlnet import ControlNetConditioningEmbedding, BaseOutput, zero_module
from diffusers.models.embeddings import CombinedTimestepTextProjEmbeddings, FluxPosEmbed, Timesteps, TimestepEmbedding
from diffusers.models.modeling_outputs import Transformer2DModelOutput


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class TimestepProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim, **kwargs):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timestep, *args, **kwargs):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(timestep.dtype))  # (N, D)

        return timesteps_emb


class FluxControlNetXSModel(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        patch_size: int = 1,
        conditioning_channels: int = 3,
        in_channels: int = 64,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 64,
        num_attention_heads: int = 4,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = None,
        axes_dims_rope: List[int] = (8, 28, 28),
        base_attention_head_dim: int = 128,
        base_num_attention_heads: int = 24,
    ):
        super().__init__()
        self.out_channels = in_channels
        self.inner_dim = num_attention_heads * attention_head_dim
        self.base_inner_dim = base_num_attention_heads * base_attention_head_dim

        self.pos_embed = FluxPosEmbed(theta=10000, axes_dim=axes_dims_rope)
        if pooled_projection_dim is not None:
            self.time_text_embed = CombinedTimestepTextProjEmbeddings(
                embedding_dim=self.inner_dim, pooled_projection_dim=pooled_projection_dim
            )
        else:
            self.time_text_embed = TimestepProjEmbeddings(embedding_dim=self.inner_dim)

        self.x_embedder = nn.Linear(in_channels, self.inner_dim)
        self.context_embedder = nn.Linear(joint_attention_dim, self.inner_dim)

        self.controlnet_cond_embedding = ControlNetConditioningEmbedding(
            conditioning_embedding_channels=16,
            conditioning_channels=conditioning_channels,
        )
        self.controlnet_x_embedder = nn.Linear(in_channels, self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                FluxTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                )
                for _ in range(num_layers)
            ]
        )

        self.single_transformer_blocks = nn.ModuleList(
            [
                FluxSingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                )
                for _ in range(num_single_layers)
            ]
        )

        # controlnet_proj
        self.ctrl_proj = nn.ModuleList([])
        for _ in range(num_layers):
            self.ctrl_proj.append(zero_module(nn.Linear(self.inner_dim, self.base_inner_dim)))

        self.single_ctrl_proj = nn.ModuleList([])
        for _ in range(num_single_layers):
            self.single_ctrl_proj.append(zero_module(nn.Linear(self.inner_dim, self.base_inner_dim)))

        # base_proj
        self.base_proj = nn.ModuleList([])
        for _ in range(num_layers):
            self.base_proj.append(zero_module(
                nn.Linear(self.base_inner_dim + self.inner_dim, self.inner_dim)))

        self.single_base_proj = nn.ModuleList([])
        for _ in range(num_single_layers):
            self.single_base_proj.append(zero_module(
                nn.Linear(self.base_inner_dim + self.inner_dim, self.inner_dim)))

        self.gradient_checkpointing = False

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self):
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    @staticmethod
    def _pack_latents(latents):
        batch_size, num_channels_latents, height, width = latents.shape
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

        return latents

    @staticmethod
    def _unpack_latents(latents, height, width):
        batch_size, num_patches, channels = latents.shape

        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // (2 * 2), height, width)

        return latents

    @classmethod
    def from_transformer(
        cls,
        transformer,
        conditioning_channels: int = 3,
        attention_head_dim: int = 64,
        num_attention_heads: int = 4,
        axes_dims_rope=(8, 28, 28),
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = None,
    ):
        config = {
            "conditioning_channels": conditioning_channels,
            "in_channels": transformer.config["in_channels"],
            "num_layers": transformer.config["num_layers"],
            "num_single_layers": transformer.config["num_single_layers"],
            "attention_head_dim": attention_head_dim,
            "num_attention_heads": num_attention_heads,
            "axes_dims_rope": axes_dims_rope,
            "joint_attention_dim": joint_attention_dim,
            "pooled_projection_dim": pooled_projection_dim,
            "base_attention_head_dim": transformer.config["attention_head_dim"],
            "base_num_attention_heads": transformer.config["num_attention_heads"],
        }

        return cls(**config)

    def forward(
        self,
        base_model: FluxTransformer2DModel,
        hidden_states: torch.Tensor,
        controlnet_cond: torch.Tensor,
        conditioning_scale: float = 1.0,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        image_embeds: torch.Tensor = None,
        pooled_image_embeds: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        added_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        base_hidden_states: torch.Tensor = None,
        return_dict: bool = True,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:

        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(base_model, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )
        # 0.1 base x_embedder
        base_hidden_states = base_hidden_states if base_hidden_states is not None else hidden_states
        h_base = base_model.x_embedder(base_hidden_states)

        # 0.2 control add
        h_ctrl = self.x_embedder(hidden_states)
        controlnet_cond = self.controlnet_cond_embedding(controlnet_cond)
        controlnet_cond = self._pack_latents(controlnet_cond)
        h_ctrl = h_ctrl + self.controlnet_x_embedder(controlnet_cond)

        # 1. time and text embedding
        timestep = timestep.to(hidden_states.dtype)
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        else:
            guidance = None

        temb_base = (
            base_model.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else base_model.time_text_embed(timestep, guidance, pooled_projections)
        )
        cemb_base = base_model.context_embedder(encoder_hidden_states)

        temb_ctrl = self.time_text_embed(timestep, pooled_projections)
        cemb_ctrl = self.context_embedder(encoder_hidden_states)

        # 2. image and text rotary_emb
        ids = torch.cat((txt_ids, img_ids), dim=0)
        rotary_emb = base_model.pos_embed(ids)
        rotary_emb_ctrl = self.pos_embed(ids)

        txt_seq_len = None
        added_rotary_emb = None
        if added_ids is not None:
            added_rotary_emb = base_model.pos_embed(torch.cat((txt_ids, img_ids, added_ids), dim=0))

        if image_embeds is not None:
            if hasattr(base_model, "ip_image_proj"):
                image_embeds = base_model.ip_image_proj(image_embeds)

            if added_ids is None:
                txt_seq_len = encoder_hidden_states.shape[1]

            if pooled_image_embeds is not None and hasattr(base_model, "pooled_im_embedder"):
                pooled_image_embeds = base_model.pooled_im_embedder(pooled_image_embeds)
                temb_base = temb_base + pooled_image_embeds

        # double stream blocks
        for i, (block, c_block) in enumerate(zip(base_model.transformer_blocks, self.transformer_blocks)):
            # base model
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                cemb_base, h_base = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    h_base,
                    cemb_base,
                    temb_base,
                    rotary_emb,
                    image_embeds,
                    txt_seq_len,
                    added_rotary_emb,
                    **ckpt_kwargs,
                )

            else:
                cemb_base, h_base = block(
                    hidden_states=h_base,
                    encoder_hidden_states=cemb_base,
                    temb=temb_base,
                    image_rotary_emb=rotary_emb,
                    image_embeds=image_embeds,
                    txt_seq_len=txt_seq_len,
                    added_rotary_emb=added_rotary_emb,
                )

            if conditioning_scale > 0.:
                # control model
                cemb_ctrl, h_ctrl = c_block(
                    hidden_states=h_ctrl,
                    encoder_hidden_states=cemb_ctrl,
                    temb=temb_ctrl,
                    image_rotary_emb=rotary_emb_ctrl,
                )
                # control projection
                h_ctrl_proj = self.ctrl_proj[i](h_ctrl)
                h_ctrl = self.base_proj[i](torch.cat([h_base, h_ctrl], dim=2))
                h_base = h_base + conditioning_scale * h_ctrl_proj

        # single stream blocks
        h_base = torch.cat([cemb_base, h_base], dim=1)
        h_ctrl = torch.cat([cemb_ctrl, h_ctrl], dim=1)
        c_len = cemb_base.shape[1]
        for i, (block, c_block) in enumerate(zip(base_model.single_transformer_blocks, self.single_transformer_blocks)):
            # base model
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                h_base = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    h_base,
                    temb_base,
                    rotary_emb,
                    image_embeds,
                    txt_seq_len,
                    added_rotary_emb,
                    **ckpt_kwargs,
                )

            else:
                h_base = block(
                    hidden_states=h_base,
                    temb=temb_base,
                    image_rotary_emb=rotary_emb,
                    image_embeds=image_embeds,
                    txt_seq_len=txt_seq_len,
                    added_rotary_emb=added_rotary_emb,
                )

            if conditioning_scale > 0.:
                # control model
                h_ctrl = c_block(
                    hidden_states=h_ctrl,
                    temb=temb_ctrl,
                    image_rotary_emb=rotary_emb_ctrl,
                )
                # control projection
                cemb_base, h_base = h_base[:, :c_len], h_base[:, c_len:]
                cemb_ctrl, h_ctrl = h_ctrl[:, :c_len], h_ctrl[:, c_len:]
                h_ctrl_proj = self.single_ctrl_proj[i](h_ctrl)
                h_ctrl = self.single_base_proj[i](torch.cat([h_base, h_ctrl], dim=2))
                h_base = h_base + conditioning_scale * h_ctrl_proj

                h_base = torch.cat([cemb_base, h_base], dim=1)
                h_ctrl = torch.cat([cemb_ctrl, h_ctrl], dim=1)

        h_base = h_base[:, c_len:]
        h_base = base_model.norm_out(h_base, temb_base)
        output = base_model.proj_out(h_base)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(base_model, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
