# Copyright 2024 Black Forest Labs, The HuggingFace Team and The InstantX Team. All rights reserved.
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


from collections import defaultdict
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import (
    Attention,
    AttentionProcessor,
    FluxAttnProcessor2_0,
    FusedFluxAttnProcessor2_0,
)
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import (
    AdaLayerNormContinuous,
    AdaLayerNormZero,
    AdaLayerNormZeroSingle,
    RMSNorm,
)
from diffusers.utils import (
    USE_PEFT_BACKEND,
    logging,
    scale_lora_layers,
    unscale_lora_layers
)
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.embeddings import (
    CombinedTimestepGuidanceTextProjEmbeddings,
    CombinedTimestepTextProjEmbeddings,
    FluxPosEmbed,
    IPAdapterPlusImageProjection,
    apply_rotary_emb,
    Timesteps,
    TimestepEmbedding,
)
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils.torch_utils import is_compiled_module


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


class IPAdapterFluxAttnProcessor2_0(torch.nn.Module):
    """ IP-Adapter Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self, embed_dims, hidden_dims, scale=1.0):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("IPAdapterFluxAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.scale = scale

        self.to_k_ip = nn.Linear(embed_dims, hidden_dims, bias=False)
        self.to_v_ip = nn.Linear(embed_dims, hidden_dims, bias=False)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        txt_seq_len: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        # IP-Adapter
        if image_embeds is not None and self.scale > 0:
            ip_query = query[:, :, txt_seq_len:]
            ip_key = self.to_k_ip(image_embeds)
            ip_value = self.to_v_ip(image_embeds)

            ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            current_ip_hidden_states = F.scaled_dot_product_attention(
                ip_query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            current_ip_hidden_states = current_ip_hidden_states.transpose(1, 2).reshape(
                batch_size, -1, attn.heads * head_dim
            ).to(query.dtype)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # Add IP-Adapter to hidden_states
        if image_embeds is not None and self.scale > 0:
            ip_encoder_hidden_states, im_hidden_states = (
                hidden_states[:, :txt_seq_len],
                hidden_states[:, txt_seq_len:],
            )
            im_hidden_states = im_hidden_states + self.scale * current_ip_hidden_states
            hidden_states = torch.cat([ip_encoder_hidden_states, im_hidden_states], dim=1)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


class IPAdapterFluxSelfAttnProcessor2_0(torch.nn.Module):
    """ IP-Adapter Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self, embed_dims, hidden_dims, dim_head=128, use_refer=False):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("IPAdapterFluxSelfAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.use_refer = use_refer
        if not use_refer:
            self.to_k_ip = nn.Linear(embed_dims, hidden_dims, bias=True)
            self.to_v_ip = nn.Linear(embed_dims, hidden_dims, bias=True)
            self.norm_k_ip = RMSNorm(dim_head, eps=1e-5)

        self.k_bank = None
        self.v_bank = None

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        added_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        # IP-Adapter self-attn
        if added_rotary_emb is not None:
            if image_embeds is not None:
                ip_key = self.to_k_ip(image_embeds)
                ip_value = self.to_v_ip(image_embeds)

                ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

                ip_key = self.norm_k_ip(ip_key)
            else:
                ip_key = self.k_bank
                ip_value = self.v_bank

            # attention
            key = torch.cat([key, ip_key], dim=2)
            value = torch.cat([value, ip_value], dim=2)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            if added_rotary_emb is None:
                key = apply_rotary_emb(key, image_rotary_emb)
            else:
                key = apply_rotary_emb(key, added_rotary_emb)

        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


@maybe_allow_in_graph
class FluxSingleTransformerBlock(nn.Module):
    r"""
    A Transformer block following the MMDiT architecture, introduced in Stable Diffusion 3.

    Reference: https://arxiv.org/abs/2403.03206

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        context_pre_only (`bool`): Boolean to determine if we should add some blocks associated with the
            processing of `context` conditions.
    """

    def __init__(self, dim, num_attention_heads, attention_head_dim, mlp_ratio=4.0):
        super().__init__()
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm = AdaLayerNormZeroSingle(dim)
        self.proj_mlp = nn.Linear(dim, self.mlp_hidden_dim)
        self.act_mlp = nn.GELU(approximate="tanh")
        self.proj_out = nn.Linear(dim + self.mlp_hidden_dim, dim)

        processor = FluxAttnProcessor2_0()
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            processor=processor,
            qk_norm="rms_norm",
            eps=1e-6,
            pre_only=True,
        )

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        image_rotary_emb=None,
        image_embeds=None,
        txt_seq_len=None,
        added_rotary_emb=None,
    ):
        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))

        kwargs = {}
        if image_embeds is not None:
            kwargs['image_embeds'] = image_embeds
        if txt_seq_len is not None:
            kwargs['txt_seq_len'] = txt_seq_len
        if added_rotary_emb is not None:
            kwargs['added_rotary_emb'] = added_rotary_emb

        # Attention.
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **kwargs
        )

        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        gate = gate.unsqueeze(1)
        hidden_states = gate * self.proj_out(hidden_states)
        hidden_states = residual + hidden_states
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return hidden_states


@maybe_allow_in_graph
class FluxTransformerBlock(nn.Module):
    r"""
    A Transformer block following the MMDiT architecture, introduced in Stable Diffusion 3.

    Reference: https://arxiv.org/abs/2403.03206

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        context_pre_only (`bool`): Boolean to determine if we should add some blocks associated with the
            processing of `context` conditions.
    """

    def __init__(self, dim, num_attention_heads, attention_head_dim, qk_norm="rms_norm", eps=1e-6):
        super().__init__()

        self.norm1 = AdaLayerNormZero(dim)

        self.norm1_context = AdaLayerNormZero(dim)

        if hasattr(F, "scaled_dot_product_attention"):
            processor = FluxAttnProcessor2_0()
        else:
            raise ValueError(
                "The current PyTorch version does not support the `scaled_dot_product_attention` function."
            )
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            processor=processor,
            qk_norm=qk_norm,
            eps=eps,
        )

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_context = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        image_rotary_emb=None,
        image_embeds=None,
        txt_seq_len=None,
        added_rotary_emb=None,
    ):
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )

        kwargs = {}
        if image_embeds is not None:
            kwargs['image_embeds'] = image_embeds
        if txt_seq_len is not None:
            kwargs['txt_seq_len'] = txt_seq_len
        if added_rotary_emb is not None:
            kwargs['added_rotary_emb'] = added_rotary_emb

        # Attention.
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **kwargs
        )

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output

        # Process attention outputs for the `encoder_hidden_states`.

        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states


# for selective op activation checkpointing
_save_list = {
    torch.ops.aten.mm.default,
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops._c10d_functional.reduce_scatter_tensor.default,
}


def apply_custom_gc_to_transformer_block(module: nn.Module):
    from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts
    def _get_custom_policy(meta):
        def _custom_policy(ctx, func, *args, **kwargs):
            mode = "recompute" if ctx.is_recompute else "forward"
            mm_count_key = f"{mode}_mm_count"
            if func == torch.ops.aten.mm.default:
                meta[mm_count_key] += 1
            # Saves output of all compute ops, except every second mm
            to_save = func in _save_list and not (
                func == torch.ops.aten.mm.default and meta[mm_count_key] % 2 == 0
            )
            return (
                CheckpointPolicy.MUST_SAVE
                if to_save
                else CheckpointPolicy.PREFER_RECOMPUTE
            )

        return _custom_policy

    def selective_checkpointing_context_fn():
        meta = defaultdict(int)
        return create_selective_checkpoint_contexts(_get_custom_policy(meta))

    return checkpoint_wrapper(
        module,
        context_fn=selective_checkpointing_context_fn,
        preserve_rng_state=False,
    )


class FluxTransformer2DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    """
    The Transformer model introduced in Flux.

    Reference: https://blackforestlabs.ai/announcing-black-forest-labs/

    Parameters:
        patch_size (`int`): Patch size to turn the input data into small patches.
        in_channels (`int`, *optional*, defaults to 16): The number of channels in the input.
        num_layers (`int`, *optional*, defaults to 18): The number of layers of MMDiT blocks to use.
        num_single_layers (`int`, *optional*, defaults to 18): The number of layers of single DiT blocks to use.
        attention_head_dim (`int`, *optional*, defaults to 64): The number of channels in each head.
        num_attention_heads (`int`, *optional*, defaults to 18): The number of heads to use for multi-head attention.
        joint_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        pooled_projection_dim (`int`): Number of dimensions to use when projecting the `pooled_projections`.
        guidance_embeds (`bool`, defaults to False): Whether to use guidance embeddings.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["FluxTransformerBlock", "FluxSingleTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 64,
        out_channels: Optional[int] = None,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = False,
        axes_dims_rope: Tuple[int] = (16, 56, 56),
        condition_embeds: bool = False,
        reference_embeds: bool = False,
    ):
        super().__init__()
        self.out_channels = out_channels or in_channels
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim

        self.pos_embed = FluxPosEmbed(theta=10000, axes_dim=axes_dims_rope)

        if self.config.guidance_embeds:
            self.time_text_embed = CombinedTimestepGuidanceTextProjEmbeddings(
                embedding_dim=self.inner_dim, pooled_projection_dim=self.config.pooled_projection_dim
            )
        elif self.config.pooled_projection_dim is not None:
            self.time_text_embed = CombinedTimestepTextProjEmbeddings(
                embedding_dim=self.inner_dim, pooled_projection_dim=self.config.pooled_projection_dim
            )
        else:
            self.time_text_embed = TimestepProjEmbeddings(embedding_dim=self.inner_dim)

        self.context_embedder = nn.Linear(self.config.joint_attention_dim, self.inner_dim)
        self.x_embedder = nn.Linear(self.config.in_channels, self.inner_dim)
        if condition_embeds:
            self.condition_embedder = nn.Linear(self.config.in_channels, self.inner_dim)
        if reference_embeds:
            self.reference_embedder = nn.Linear(self.config.in_channels, self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                FluxTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                )
                for i in range(self.config.num_layers)
            ]
        )

        self.single_transformer_blocks = nn.ModuleList(
            [
                FluxSingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                )
                for i in range(self.config.num_single_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

        self.gradient_checkpointing = False

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
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

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.fuse_qkv_projections with FusedAttnProcessor2_0->FusedFluxAttnProcessor2_0
    def fuse_qkv_projections(self):
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query, key, value)
        are fused. For cross-attention modules, key and value projection matrices are fused.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>
        """
        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError("`fuse_qkv_projections()` is not supported for models having added KV projections.")

        self.original_attn_processors = self.attn_processors

        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

        self.set_attn_processor(FusedFluxAttnProcessor2_0())

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.unfuse_qkv_projections
    def unfuse_qkv_projections(self):
        """Disables the fused QKV projection if enabled.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>

        """
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

    def set_ip_adapter_scale(self, scale):
        for attn_name, attn_processor in self.attn_processors.items():
            if isinstance(attn_processor, IPAdapterFluxAttnProcessor2_0):
                    attn_processor.scale = scale

    def _compile_transformer_block(self):
        for name, transformer_block in self.transformer_blocks.named_children():
            transformer_block = torch.compile(transformer_block, fullgraph=True)
            self.transformer_blocks.register_module(name, transformer_block)

        for name, transformer_block in self.single_transformer_blocks.named_children():
            transformer_block = torch.compile(transformer_block, fullgraph=True)
            self.single_transformer_blocks.register_module(name, transformer_block)

        logger.info("Compiled each block with torch.compile")

    def _enable_custom_gradient_checkpointing(self):
        for name, transformer_block in self.transformer_blocks.named_children():
            transformer_block = apply_custom_gc_to_transformer_block(transformer_block)
            self.transformer_blocks.register_module(name, transformer_block)

        for name, transformer_block in self.single_transformer_blocks.named_children():
            transformer_block = apply_custom_gc_to_transformer_block(transformer_block)
            self.single_transformer_blocks.register_module(name, transformer_block)

        logger.info("Applied custom gradient checkpointing to the model")

    def _init_ip_adapter_plus(self,
                              num_image_text_embeds=32,
                              embed_dims=1024,
                              num_attention_heads=16,
                              state_dict=None,
                              scale=1.0,
                              alter_x_embedder=False,
                              with_pooled_embedder=False,
                              use_self_attn=False,
                              use_refer=False):
        # set ip-adapter cross-attention processors & load state_dict
        attn_procs = {}
        for name, module in self.attn_processors.items():
            if use_self_attn:
                attn_procs[name] = IPAdapterFluxSelfAttnProcessor2_0(
                    embed_dims=embed_dims,
                    hidden_dims=self.inner_dim,
                    dim_head=self.config.attention_head_dim,
                    use_refer=use_refer,
                ).to(dtype=self.dtype, device=self.device)
            else:
                attn_procs[name] = IPAdapterFluxAttnProcessor2_0(
                    embed_dims=embed_dims,
                    hidden_dims=self.inner_dim,
                    scale=scale,
                ).to(dtype=self.dtype, device=self.device)
        self.set_attn_processor(attn_procs)

        if state_dict is not None and 'ip_image_proj.latents' in state_dict:
            num_image_text_embeds = state_dict['ip_image_proj.latents'].shape[1]

        if num_image_text_embeds > 0:
            # IP-Adapter Image Projection layers
            self.ip_image_proj = IPAdapterPlusImageProjection(
                embed_dims=embed_dims,
                output_dims=embed_dims,
                hidden_dims=embed_dims,
                heads=num_attention_heads,
                num_queries=num_image_text_embeds,
            ).to(device=self.device, dtype=self.dtype)

        # concat reference latents
        if alter_x_embedder:
            # concat reference image
            x_embedder_weights = self.x_embedder.weight.data
            x_embedder_bias = self.x_embedder.bias.data
            self.x_embedder = torch.nn.Linear(
                self.config.in_channels * 2, self.inner_dim)
            for p in self.x_embedder.parameters():
                nn.init.zeros_(p)

            self.x_embedder.weight.data[:, :self.config.in_channels] = x_embedder_weights
            self.x_embedder.bias.data = x_embedder_bias
            self.x_embedder = self.x_embedder.to(device=self.device, dtype=self.dtype)

        # pooled image embedding
        if with_pooled_embedder:
            self.pooled_im_embedder = torch.nn.Linear(embed_dims, self.inner_dim)

        # load model `state_dict`
        if state_dict is not None:
            self.load_state_dict(state_dict, strict=False)

    def _init_image_variation(self,
                              joint_attention_dim=1024,
                              pooled_projection_dim=None,
                              state_dict=None,
                              alter_x_embedder=False):
        self.register_to_config(guidance_embeds=False)
        self.register_to_config(joint_attention_dim=joint_attention_dim)
        self.register_to_config(pooled_projection_dim=pooled_projection_dim)
        text_time_guidance_cls = (
            CombinedTimestepTextProjEmbeddings if pooled_projection_dim else TimestepProjEmbeddings
        )
        time_text_embed = text_time_guidance_cls(
            embedding_dim=self.inner_dim, pooled_projection_dim=pooled_projection_dim
        )

        time_text_embed.timestep_embedder.load_state_dict(self.time_text_embed.timestep_embedder.state_dict())

        self.time_text_embed = time_text_embed.to(device=self.device, dtype=self.dtype)

        self.context_embedder = nn.Linear(joint_attention_dim, self.inner_dim).to(device=self.device, dtype=self.dtype)

        # concat reference latents
        if alter_x_embedder:
            # concat reference image
            x_embedder_weights = self.x_embedder.weight.data
            x_embedder_bias = self.x_embedder.bias.data

            self.x_embedder = torch.nn.Linear(
                self.config.in_channels * 2, self.inner_dim)
            for p in self.x_embedder.parameters():
                nn.init.zeros_(p)

            self.x_embedder.weight.data[:, :self.config.in_channels] = x_embedder_weights
            self.x_embedder.bias.data = x_embedder_bias
            self.x_embedder = self.x_embedder.to(device=self.device, dtype=self.dtype)
            # self.register_to_config(in_channels=self.config.in_channels * 2)

        if state_dict is not None:
            self.load_state_dict(state_dict, strict=False)

    def _init_fill_x_embedder(self, state_dict=None,):
        # concat pose image latents
        x_embedder_weights = self.x_embedder.weight.data
        x_embedder_bias = self.x_embedder.bias.data

        self.x_embedder = torch.nn.Linear(
            self.config.in_channels + 64, self.inner_dim)
        for p in self.x_embedder.parameters():
            nn.init.zeros_(p)

        self.x_embedder.weight.data[:, :self.config.in_channels] = x_embedder_weights
        self.x_embedder.bias.data = x_embedder_bias
        self.x_embedder = self.x_embedder.to(device=self.device, dtype=self.dtype)
        self.register_to_config(in_channels=self.config.in_channels + 64)

        if state_dict is not None:
            self.load_state_dict(state_dict, strict=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
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
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        return_dict: bool = True,
        condition_embeds: torch.Tensor = None,
        conditioning_scale: float = 1.0,
        reference_embeds: torch.Tensor = None,
        **kwargs
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        """
        The [`FluxTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.FloatTensor` of shape `(batch_size, projection_dim)`): Embeddings projected
                from the embeddings of input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            block_controlnet_hidden_states: (`list` of `torch.Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        hidden_states = self.x_embedder(hidden_states)
        if condition_embeds is not None:
            condition_embeds = self.condition_embedder(condition_embeds)
            hidden_states = hidden_states + condition_embeds * conditioning_scale

        timestep = timestep.to(hidden_states.dtype)
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        else:
            guidance = None

        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        if reference_embeds is not None:
            reference_embeds = self.reference_embedder(reference_embeds)
            encoder_hidden_states = torch.cat([encoder_hidden_states, reference_embeds], dim=1)

        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            img_ids = img_ids[0]

        image_rotary_emb = self.pos_embed(torch.cat((txt_ids, img_ids), dim=0))

        txt_seq_len = None
        added_rotary_emb = None
        if added_ids is not None:
            added_rotary_emb = self.pos_embed(torch.cat((txt_ids, img_ids, added_ids), dim=0))

        if image_embeds is not None:
            if hasattr(self, "ip_image_proj"):
                image_embeds = self.ip_image_proj(image_embeds)

            if added_ids is None:
                txt_seq_len = encoder_hidden_states.shape[1]

            if pooled_image_embeds is not None and hasattr(self, "pooled_im_embedder"):
                pooled_image_embeds = self.pooled_im_embedder(pooled_image_embeds)
                temb = temb + pooled_image_embeds

        for index_block, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    image_embeds,
                    txt_seq_len,
                    added_rotary_emb,
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    image_embeds=image_embeds,
                    txt_seq_len=txt_seq_len,
                    added_rotary_emb=added_rotary_emb,
                )

            # controlnet residual
            if controlnet_block_samples is not None:
                interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                interval_control = int(np.ceil(interval_control))
                hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        for index_block, block in enumerate(self.single_transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    temb,
                    image_rotary_emb,
                    image_embeds,
                    txt_seq_len,
                    added_rotary_emb,
                )
            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    image_embeds=image_embeds,
                    txt_seq_len=txt_seq_len,
                    added_rotary_emb=added_rotary_emb,
                )

            # controlnet residual
            if controlnet_single_block_samples is not None:
                interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
                interval_control = int(np.ceil(interval_control))
                hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
                    hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                    + controlnet_single_block_samples[index_block // interval_control]
                )

        hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
