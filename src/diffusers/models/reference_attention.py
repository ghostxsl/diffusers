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

from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.unet_3d_blocks import TransformerTemporalModel


def torch_attn_dfs(model: torch.nn.Module, prefix=""):
    result = []
    for name, child in model.named_children():
        if name == 'motion_modules' or isinstance(child, TransformerTemporalModel):
            continue
        elif isinstance(child, BasicTransformerBlock):
            result.append([prefix + f".{name}", child])
        else:
            result += torch_attn_dfs(child, prefix=prefix + f".{name}")
    return result


class ReferenceAttentionControl(object):
    def __init__(self,
                 unet,
                 mode,
                 fusion_blocks="full",
                 ):
        # Modify self attention
        self.unet = unet
        assert mode in ["read", "write"]
        assert fusion_blocks in ["midup", "full"]
        self.fusion_blocks = fusion_blocks
        self.register_reference_hooks(mode)

    def register_reference_hooks(self, mode):
        MODE = mode

        def hacked_basic_transformer_inner_forward(
                self,
                hidden_states: torch.FloatTensor,
                attention_mask: Optional[torch.FloatTensor] = None,
                encoder_hidden_states: Optional[torch.FloatTensor] = None,
                encoder_attention_mask: Optional[torch.FloatTensor] = None,
                timestep: Optional[torch.LongTensor] = None,
                cross_attention_kwargs: Dict[str, Any] = None,
                class_labels: Optional[torch.LongTensor] = None,
                # video_length=None,
        ):
            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero:
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                    hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
            else:
                norm_hidden_states = self.norm1(hidden_states)

            # 1. Self-Attention
            cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
            if self.only_cross_attention:
                attn_output = self.attn1(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                    attention_mask=attention_mask,
                    **cross_attention_kwargs,
                )
            else:
                if MODE == "write":
                    self.bank.append(norm_hidden_states.clone())
                    if hasattr(self, 'is_final_block') and self.is_final_block:
                        return norm_hidden_states
                    attn_output = self.attn1(
                        norm_hidden_states,
                        encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                        attention_mask=attention_mask,
                        **cross_attention_kwargs,
                    )
                elif MODE == "read":
                    if norm_hidden_states.shape[0] != self.bank[0].shape[0]:
                        self.bank[0] = self.bank[0].repeat_interleave(
                            repeats=norm_hidden_states.shape[0], dim=0)
                    hidden_states = self.attn1(norm_hidden_states,
                                               encoder_hidden_states=torch.cat(
                                                   [norm_hidden_states] + self.bank, dim=1),
                                               attention_mask=attention_mask) + hidden_states

                    self.bank.clear()
                    if self.attn2 is not None:
                        # Cross-Attention
                        norm_hidden_states = (
                            self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(
                                hidden_states)
                        )
                        hidden_states = (
                                self.attn2(
                                    norm_hidden_states, encoder_hidden_states=encoder_hidden_states,
                                    attention_mask=attention_mask
                                )
                                + hidden_states
                        )

                    # Feed-forward
                    hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states

                    # Temporal-Attention
                    # if self.unet_use_temporal_attention:
                    #     d = hidden_states.shape[1]
                    #     hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=video_length)
                    #     norm_hidden_states = (
                    #         self.norm_temp(hidden_states, timestep) if self.use_ada_layer_norm else self.norm_temp(
                    #             hidden_states)
                    #     )
                    #     hidden_states = self.attn_temp(norm_hidden_states) + hidden_states
                    #     hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)

                    return hidden_states

            if self.use_ada_layer_norm_zero:
                attn_output = gate_msa.unsqueeze(1) * attn_output
            hidden_states = attn_output + hidden_states

            if self.attn2 is not None:
                norm_hidden_states = (
                    self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
                )

                # 2. Cross-Attention
                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    **cross_attention_kwargs,
                )
                hidden_states = attn_output + hidden_states

            # 3. Feed-forward
            norm_hidden_states = self.norm3(hidden_states)

            if self.use_ada_layer_norm_zero:
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

            ff_output = self.ff(norm_hidden_states)

            if self.use_ada_layer_norm_zero:
                ff_output = gate_mlp.unsqueeze(1) * ff_output

            hidden_states = ff_output + hidden_states

            return hidden_states

        if self.fusion_blocks == "midup":
            attn_modules = torch_attn_dfs(self.unet.mid_block) + torch_attn_dfs(self.unet.up_blocks)
        elif self.fusion_blocks == "full":
            attn_modules = torch_attn_dfs(self.unet)

        for i, (name, module) in enumerate(attn_modules):
            module._original_inner_forward = module.forward
            module.forward = hacked_basic_transformer_inner_forward.__get__(module, BasicTransformerBlock)
            module.bank = []

    def update(self, writer, dtype=torch.float32):
        if self.fusion_blocks == "midup":
            reader_attn_modules = torch_attn_dfs(self.unet.mid_block) + torch_attn_dfs(self.unet.up_blocks)
            writer_attn_modules = torch_attn_dfs(writer.unet.mid_block) + torch_attn_dfs(writer.unet.up_blocks)
        elif self.fusion_blocks == "full":
            reader_attn_modules = torch_attn_dfs(self.unet)
            writer_attn_modules = torch_attn_dfs(writer.unet)

        for (_, r), (_, w) in zip(reader_attn_modules, writer_attn_modules):
            r.bank = [v.clone().to(dtype) for v in w.bank]
            # w.bank.clear()

    def clear(self):
        if self.fusion_blocks == "midup":
            attn_modules = torch_attn_dfs(self.unet.mid_block) + torch_attn_dfs(self.unet.up_blocks)
        elif self.fusion_blocks == "full":
            attn_modules = torch_attn_dfs(self.unet)

        for name, module in attn_modules:
            module.bank.clear()
