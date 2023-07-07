# Copyright (c) wilson.xu. All rights reserved.
import math
import os
import weakref
from collections import OrderedDict
import torch
from safetensors.torch import load_file, save_file

from ..utils import logging


__all__ = ['ModelEMA']
logger = logging.get_logger(__name__)


class ModelEMA(object):
    """
    Exponential Weighted Average for Deep Neural Networks

    Args:
        model (nn.Module): torch model.
        decay (float):  The decay used for updating ema parameter.
            EMA's parameter are updated with the formula:
           `ema_param = decay * ema_param + (1 - decay) * cur_param`.
            Default: 0.9998.
        ema_decay_type (str): EMA type in ['simple', 'exponential'],
            Default: 'exponential'.
    """

    def __init__(self, model, decay=0.9998, ema_decay_type='exponential'):
        self.decay = decay
        self.ema_decay_type = ema_decay_type
        self.step = 0

        self.state_dict = OrderedDict()
        model_state = model.state_dict(keep_vars=True)
        for k, v in model_state.items():
            if v.requires_grad:
                self.state_dict[k] = torch.zeros_like(v)
            else:
                self.state_dict[k] = v.detach()

        self._model_state = {
            k: weakref.ref(v) for k, v in model_state.items()}

    @torch.no_grad()
    def update(self, model=None):
        if self.ema_decay_type == 'exponential':
            decay = self.decay * (1 - math.exp(-(self.step + 1) / 2000))
        else:
            decay = self.decay

        if model is not None:
            model_dict = model.state_dict()
        else:
            model_dict = {k: p() for k, p in self._model_state.items()}
            assert all(
                [v is not None for _, v in model_dict.items()]), 'python gc.'

        for k, v in self.state_dict.items():
            if k not in model_dict:
                continue

            if model_dict[k].requires_grad:
                v = decay * v + (1 - decay) * model_dict[k].detach()
                self.state_dict[k] = v

        self.step += 1

    def resume(self, state_dict, step=0, device=torch.device("cuda")):
        unexpected_keys = []
        for k, v in state_dict.items():
            if k in self.state_dict:
                self.state_dict[k] = v.to(device)
            else:
                unexpected_keys.append(k)
        self.step = step
        if len(unexpected_keys) > 0:
            print(unexpected_keys)

    def save_model(self,
                   save_directory,
                   safe_serialization=True,
                   weights_name="diffusion_pytorch_model.safetensors"):
        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return
        os.makedirs(save_directory, exist_ok=True)

        # Save the model
        self.state_dict["step"] = torch.tensor(self.step)
        if safe_serialization:
            save_file(
                self.state_dict, os.path.join(save_directory, weights_name), metadata={"format": "pt"}
            )
        else:
            torch.save(self.state_dict, os.path.join(save_directory, weights_name))
        self.state_dict.pop("step")

        logger.info(f"Model weights saved in {os.path.join(save_directory, weights_name)}")
