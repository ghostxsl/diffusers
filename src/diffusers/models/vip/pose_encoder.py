# Copyright (c) wilson.xu. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.embeddings import Timesteps, TimestepEmbedding
from diffusers.models.normalization import AdaLayerNormZeroSingle


class PoseEncoder(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
            self,
            conditioning_embedding_channels: int = 3072,
            conditioning_channels: int = 3,
            block_out_channels=(16, 32, 96, 256),
    ):
        super(PoseEncoder, self).__init__()

        self.conv_in = nn.Conv2d(conditioning_channels, block_out_channels[0], kernel_size=3, padding=1)

        self.blocks = nn.ModuleList([])
        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=block_out_channels[-1] * 4)
        self.norm = AdaLayerNormZeroSingle(block_out_channels[-1] * 4)

        self.cond_embedder = nn.Linear(block_out_channels[-1] * 4, block_out_channels[-1] * 4)
        self.cond_out = nn.Linear(block_out_channels[-1] * 4, conditioning_embedding_channels)

    @staticmethod
    def _pack_latents(latents):
        batch_size, num_channels_latents, height, width = latents.shape
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

        return latents

    def forward(self, conditioning, timestep):
        # condition encode
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)
        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)
        hidden_states = self._pack_latents(embedding)

        # time_embedding
        timesteps_proj = self.time_proj(timestep.to(conditioning.dtype))
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=conditioning.dtype))

        norm_hidden_states, gate = self.norm(hidden_states, emb=timesteps_emb)

        hidden_states = gate.unsqueeze(1) * self.cond_embedder(norm_hidden_states)
        hidden_states = self.cond_out(hidden_states)

        return hidden_states


class TimestepProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timestep):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(timestep.dtype))  # (N, D)

        return timesteps_emb


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module
