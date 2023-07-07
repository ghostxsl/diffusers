# Copyright 2023 ByteDance and/or its affiliates.
#
# Copyright (2023) MagicAnimate Authors
# Copyright 2023 The VIP Inc. AIGC team. All rights reserved.
#
# ByteDance, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from ByteDance or
# its affiliates is strictly prohibited.

import torch
from typing import Any, Dict, Optional

from .attention import BasicTransformerBlock, _chunked_feed_forward
from .unets.unet_3d_blocks import TransformerTemporalModel
from .referencenet import AttnIdentity
from .transformers.vip_transformer_2d import VIPBasicTransformerBlock
from .vip.pt_referencenet import InnerAttnIdentity
from .vip.sim_referencenet import SimAttnIdentity


def torch_attn_dfs(model, prefix=""):
    result = []
    for name, child in model.named_children():
        if name == 'motion_modules' or isinstance(child, TransformerTemporalModel):
            continue
        elif isinstance(child, (BasicTransformerBlock, VIPBasicTransformerBlock)):
            result.append([prefix + f".{name}", child])
        elif isinstance(child, (AttnIdentity, InnerAttnIdentity, SimAttnIdentity)):
            result.append([prefix + f".{name}", child])
        else:
            result += torch_attn_dfs(child, prefix=prefix + f".{name}")
    return result


class ReferenceAttentionControl(object):
    def __init__(self, unet, fusion_blocks="full", hack_type="default"):
        assert fusion_blocks in ["midup", "full", "down_flip"]

        # Modify self attention
        self.unet = unet
        self.fusion_blocks = fusion_blocks
        self.hack_type = hack_type
        self.register_reference_hooks()

    def register_reference_hooks(self):

        def hacked_basic_transformer_inner_forward(
            self,
            hidden_states: torch.FloatTensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            timestep: Optional[torch.LongTensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None,
            class_labels: Optional[torch.LongTensor] = None,
            added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        ) -> torch.FloatTensor:
            # Notice that normalization is always applied before the real computation in the following blocks.
            # 0. Self-Attention
            batch_size = hidden_states.shape[0]

            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero:
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                    hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
            elif self.use_layer_norm:
                norm_hidden_states = self.norm1(hidden_states)
            elif self.use_ada_layer_norm_continuous:
                norm_hidden_states = self.norm1(hidden_states, added_cond_kwargs["pooled_text_emb"])
            elif self.use_ada_layer_norm_single:
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                    self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
                ).chunk(6, dim=1)
                norm_hidden_states = self.norm1(hidden_states)
                norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
                norm_hidden_states = norm_hidden_states.squeeze(1)
            else:
                raise ValueError("Incorrect norm used")

            if self.pos_embed is not None:
                norm_hidden_states = self.pos_embed(norm_hidden_states)

            # 1. Prepare GLIGEN inputs
            cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
            gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

            if norm_hidden_states.shape[0] != self.bank.shape[0]:
                uc_size = norm_hidden_states.shape[0] // self.bank.shape[0]
                self.bank = self.bank.repeat_interleave(repeats=uc_size, dim=0)
            attn_output = self.attn1(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states if self.only_cross_attention\
                    else torch.cat([norm_hidden_states, self.bank], dim=1),
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )
            if self.use_ada_layer_norm_zero:
                attn_output = gate_msa.unsqueeze(1) * attn_output
            elif self.use_ada_layer_norm_single:
                attn_output = gate_msa * attn_output

            hidden_states = attn_output + hidden_states
            if hidden_states.ndim == 4:
                hidden_states = hidden_states.squeeze(1)

            # 1.5 GLIGEN Control
            if gligen_kwargs is not None:
                hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

            # 2. Cross-Attention
            if self.attn2 is not None:
                if self.use_ada_layer_norm:
                    norm_hidden_states = self.norm2(hidden_states, timestep)
                elif self.use_ada_layer_norm_zero or self.use_layer_norm:
                    norm_hidden_states = self.norm2(hidden_states)
                elif self.use_ada_layer_norm_single:
                    # For PixArt norm2 isn't applied here:
                    # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L70C1-L76C103
                    norm_hidden_states = hidden_states
                elif self.use_ada_layer_norm_continuous:
                    norm_hidden_states = self.norm2(hidden_states, added_cond_kwargs["pooled_text_emb"])
                else:
                    raise ValueError("Incorrect norm")

                if self.pos_embed is not None and self.use_ada_layer_norm_single is False:
                    norm_hidden_states = self.pos_embed(norm_hidden_states)

                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    **cross_attention_kwargs,
                )
                hidden_states = attn_output + hidden_states

            # 3. Feed-forward
            if self.use_ada_layer_norm_continuous:
                norm_hidden_states = self.norm3(hidden_states, added_cond_kwargs["pooled_text_emb"])
            elif not self.use_ada_layer_norm_single:
                norm_hidden_states = self.norm3(hidden_states)

            if self.use_ada_layer_norm_zero:
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

            if self.use_ada_layer_norm_single:
                norm_hidden_states = self.norm2(hidden_states)
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

            if self._chunk_size is not None:
                # "feed_forward_chunk_size" can be used to save memory
                ff_output = _chunked_feed_forward(
                    self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size
                )
            else:
                ff_output = self.ff(norm_hidden_states)

            if self.use_ada_layer_norm_zero:
                ff_output = gate_mlp.unsqueeze(1) * ff_output
            elif self.use_ada_layer_norm_single:
                ff_output = gate_mlp * ff_output

            hidden_states = ff_output + hidden_states
            if hidden_states.ndim == 4:
                hidden_states = hidden_states.squeeze(1)

            return hidden_states

        def hacked_vip_basic_transformer_inner_forward(
            self,
            hidden_states: torch.FloatTensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            timestep: Optional[torch.LongTensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None,
            compress_hidden_states: Optional[torch.FloatTensor] = None,
        ) -> torch.FloatTensor:
            # Notice that normalization is always applied before the real computation in the following blocks.
            # 1. Self-Attention
            norm_hidden_states = self.norm1(hidden_states)

            if self.pos_embed is not None:
                norm_hidden_states = self.pos_embed(norm_hidden_states)

            cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}

            attn_output = self.attn1(
                norm_hidden_states,
                encoder_hidden_states=self.bank if self.bank is not None else compress_hidden_states,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )

            hidden_states = attn_output + hidden_states
            if hidden_states.ndim == 4:
                hidden_states = hidden_states.squeeze(1)

            # 2. Cross-Attention
            if self.attn2 is not None:
                norm_hidden_states = self.norm2(hidden_states)

                if self.pos_embed is not None and self.norm_type != "ada_norm_single":
                    norm_hidden_states = self.pos_embed(norm_hidden_states)

                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    **cross_attention_kwargs,
                )
                hidden_states = attn_output + hidden_states

            # 3. Feed-forward
            if self._chunk_size is not None:
                # "feed_forward_chunk_size" can be used to save memory
                ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
            else:
                ff_output = self.ff(norm_hidden_states)

            hidden_states = ff_output + hidden_states
            if hidden_states.ndim == 4:
                hidden_states = hidden_states.squeeze(1)

            return hidden_states

        if self.fusion_blocks == "midup":
            attn_modules = torch_attn_dfs(self.unet.mid_block) + torch_attn_dfs(self.unet.up_blocks)
        elif self.fusion_blocks in ["full", "down_flip"]:
            attn_modules = torch_attn_dfs(self.unet)

        for i, (name, module) in enumerate(attn_modules):
            module._original_inner_forward = module.forward
            if self.hack_type == "vip":
                module.forward = hacked_vip_basic_transformer_inner_forward.__get__(module, VIPBasicTransformerBlock)
            else:
                module.forward = hacked_basic_transformer_inner_forward.__get__(module, BasicTransformerBlock)
            module.bank = None

    def update(self, referencenet, dtype=torch.float32):
        if self.fusion_blocks == "midup":
            reader_attn_modules = torch_attn_dfs(self.unet.mid_block) + torch_attn_dfs(self.unet.up_blocks)
            writer_attn_modules = torch_attn_dfs(referencenet.mid_block) + torch_attn_dfs(referencenet.up_blocks)
        elif self.fusion_blocks == "full":
            reader_attn_modules = torch_attn_dfs(self.unet)
            writer_attn_modules = torch_attn_dfs(referencenet)
        elif self.fusion_blocks == "down_flip":
            reader_down_modules = torch_attn_dfs(self.unet.down_blocks)
            reader_mid_modules = torch_attn_dfs(self.unet.mid_block)
            reader_up_modules = torch_attn_dfs(self.unet.up_blocks)

            writer_down_modules = torch_attn_dfs(referencenet.down_blocks)
            writer_mid_modules = torch_attn_dfs(referencenet.mid_block)
            # down
            for (_, r), (_, w) in zip(reader_down_modules, writer_down_modules):
                r.bank = w.bank.to(dtype)
            # mid
            for (_, r), (_, w) in zip(reader_mid_modules, writer_mid_modules):
                r.bank = w.bank.to(dtype)
            # up
            for (_, r), (_, w) in zip(reader_up_modules, writer_down_modules[::-1]):
                r.bank = w.bank.to(dtype)

            return

        for (_, r), (_, w) in zip(reader_attn_modules, writer_attn_modules):
            r.bank = w.bank.to(dtype)

    def clear(self):
        if self.fusion_blocks == "midup":
            attn_modules = torch_attn_dfs(self.unet.mid_block) + torch_attn_dfs(self.unet.up_blocks)
        elif self.fusion_blocks in ["full", "down_flip"]:
            attn_modules = torch_attn_dfs(self.unet)

        for name, module in attn_modules:
            module.bank = None
