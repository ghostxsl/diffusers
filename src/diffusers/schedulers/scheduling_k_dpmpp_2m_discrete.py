# Copyright 2023 The HuggingFace Team. All rights reserved.
# Copyright 2023 VIP AIGC Team. All rights reserved.
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
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import logging
from .scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin, SchedulerOutput


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Copied from diffusers.schedulers.scheduling_ddpm.betas_for_alpha_bar
def betas_for_alpha_bar(
    num_diffusion_timesteps,
    max_beta=0.999,
    alpha_transform_type="cosine",
):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].

    Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
    to that part of the diffusion process.


    Args:
        num_diffusion_timesteps (`int`): the number of betas to produce.
        max_beta (`float`): the maximum beta to use; use values lower than 1 to
                     prevent singularities.
        alpha_transform_type (`str`, *optional*, default to `cosine`): the type of noise schedule for alpha_bar.
                     Choose from `cosine` or `exp`

    Returns:
        betas (`np.ndarray`): the betas used by the scheduler to step the model outputs
    """
    if alpha_transform_type == "cosine":

        def alpha_bar_fn(t):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    elif alpha_transform_type == "exp":

        def alpha_bar_fn(t):
            return math.exp(t * -12.0)

    else:
        raise ValueError(f"Unsupported alpha_tranform_type: {alpha_transform_type}")

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


class KDPMPP2MDiscreteScheduler(SchedulerMixin, ConfigMixin):
    """
    Scheduler created by @crowsonkb in [k_diffusion](https://github.com/crowsonkb/k-diffusion), see:
    https://github.com/crowsonkb/k-diffusion/blob/5b3af030dd83e0297272d861c19477735d0317ec/k_diffusion/sampling.py#L585

    Scheduler inspired by DPM-Solver-2 and Algorthim 2 from Karras et al. (2022).

    [`~ConfigMixin`] takes care of storing all config attributes that are passed in the scheduler's `__init__`
    function, such as `num_train_timesteps`. They can be accessed via `scheduler.config.num_train_timesteps`.
    [`SchedulerMixin`] provides general loading and saving functionality via the [`SchedulerMixin.save_pretrained`] and
    [`~SchedulerMixin.from_pretrained`] functions.

    Args:
        num_train_timesteps (`int`): number of diffusion steps used to train the model. beta_start (`float`): the
        starting `beta` value of inference. beta_end (`float`): the final `beta` value. beta_schedule (`str`):
            the beta schedule, a mapping from a beta range to a sequence of betas for stepping the model. Choose from
            `linear` or `scaled_linear`.
        trained_betas (`np.ndarray`, optional):
            option to pass an array of betas directly to the constructor to bypass `beta_start`, `beta_end` etc.
            options to clip the variance used when adding noise to the denoised sample. Choose from `fixed_small`,
            `fixed_small_log`, `fixed_large`, `fixed_large_log`, `learned` or `learned_range`.
        prediction_type (`str`, default `epsilon`, optional):
            prediction type of the scheduler function, one of `epsilon` (predicting the noise of the diffusion
            process), `sample` (directly predicting the noisy sample`) or `v_prediction` (see section 2.4
            https://imagen.research.google/video/paper.pdf)
        timestep_spacing (`str`, default `"linspace"`):
            The way the timesteps should be scaled. Refer to Table 2. of [Common Diffusion Noise Schedules and Sample
            Steps are Flawed](https://arxiv.org/abs/2305.08891) for more information.
        steps_offset (`int`, default `0`):
            an offset added to the inference steps. You can use a combination of `offset=1` and
            `set_alpha_to_one=False`, to make the last step use step 0 for the previous alpha product, as done in
            stable diffusion.
    """

    _compatibles = [e.name for e in KarrasDiffusionSchedulers]
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        beta_schedule: str = "linear",
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        prediction_type: str = "epsilon",
        timestep_spacing: str = "linspace",
        steps_offset: int = 0,
    ):
        if trained_betas is not None:
            self.betas = torch.tensor(trained_betas, dtype=torch.float32)
        elif beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            self.betas = (
                torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
            )
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide cosine schedule
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        else:
            raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        self.sigmas_min = sigmas.min()
        self.sigmas_max = sigmas.max()
        self.log_sigmas = torch.from_numpy(np.log(sigmas).astype(np.float32))
        sigmas = np.concatenate([sigmas[::-1], [0.0]]).astype(np.float32)
        self.sigmas = torch.from_numpy(sigmas)

        # setable values
        self.num_inference_steps = None
        timesteps = np.linspace(0, num_train_timesteps - 1, num_train_timesteps, dtype=float)[::-1].copy()
        self.timesteps = torch.from_numpy(timesteps)
        self.is_scale_input_called = False

    @property
    def init_noise_sigma(self):
        # standard deviation of the initial noise distribution
        if self.config.timestep_spacing in ["linspace", "trailing"]:
            return self.sigmas.max()

        return (self.sigmas.max() ** 2 + 1) ** 0.5

    def scale_model_input(
        self, sample: torch.FloatTensor, timestep: Union[float, torch.FloatTensor]
    ) -> torch.FloatTensor:
        """
        Scales the denoising model input by `(sigma**2 + 1) ** 0.5` to match the Euler algorithm.

        Args:
            sample (`torch.FloatTensor`): input sample
            timestep (`float` or `torch.FloatTensor`): the current timestep in the diffusion chain

        Returns:
            `torch.FloatTensor`: scaled input sample
        """
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to(self.timesteps.device)
        step_index = (self.timesteps == timestep).nonzero().item()
        sigma = self.sigmas[step_index]
        sample = sample / ((sigma**2 + 1) ** 0.5)
        self.is_scale_input_called = True
        return sample

    def sigmas_to_timesteps(self, sigma, quantize=False):
        log_sigma = sigma.log()
        dists = log_sigma - self.log_sigmas[:, None]
        if quantize:
            return dists.abs().argmin(dim=0).view(sigma.shape)
        low_idx = dists.ge(0).cumsum(dim=0).argmax(dim=0).clamp(max=self.log_sigmas.shape[0] - 2)
        high_idx = low_idx + 1
        low, high = self.log_sigmas[low_idx], self.log_sigmas[high_idx]
        w = (low - log_sigma) / (low - high)
        w = w.clamp(0, 1)
        t = (1 - w) * low_idx + w * high_idx
        return t.view(sigma.shape)

    def get_sigmas_karras(self, num_inference_steps, rho=7., device='cpu'):
        """Constructs the noise schedule of Karras et al. (2022)."""
        ramp = torch.linspace(0, 1, num_inference_steps)
        min_inv_rho = self.sigmas_min ** (1 / rho)
        max_inv_rho = self.sigmas_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        timesteps = []
        for s in sigmas:
            timesteps.append(self.sigmas_to_timesteps(s))
        return torch.cat([sigmas, sigmas.new_zeros([1])]).to(device), torch.stack(timesteps).to(device)

    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        """
        Sets the timesteps used for the diffusion chain. Supporting function to be run before inference.

        Args:
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, optional):
                the device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        """
        self.num_inference_steps = num_inference_steps

        self.sigmas, self.timesteps = self.get_sigmas_karras(num_inference_steps, device=device)

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[SchedulerOutput, Tuple]:
        """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`): direct output from learned diffusion model.
            timestep (`float`): current timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                current instance of sample being created by diffusion process.
            generator (`torch.Generator`, optional): Random number generator.
            return_dict (`bool`): option for returning tuple rather than SchedulerOutput class

        Returns:
            [`~schedulers.scheduling_utils.SchedulerOutput`] or `tuple`:
            [`~schedulers.scheduling_utils.SchedulerOutput`] if `return_dict` is True, otherwise
            a `tuple`. When returning a tuple, the first element is the sample tensor.

        """

        if (
            isinstance(timestep, int)
            or isinstance(timestep, torch.IntTensor)
            or isinstance(timestep, torch.LongTensor)
        ):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `EulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if not self.is_scale_input_called:
            logger.warning(
                "The `scale_model_input` function should be called before `step` to ensure correct denoising. "
                "See `StableDiffusionPipeline` for a usage example."
            )

        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to(self.timesteps.device)

        step_index = (self.timesteps == timestep).nonzero().item()
        sigma = self.sigmas[step_index]
        # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        if self.config.prediction_type == "epsilon":
            pred_original_sample = sample - sigma * model_output
        elif self.config.prediction_type == "v_prediction":
            # * c_out + input * c_skip
            pred_original_sample = model_output * (-sigma / (sigma ** 2 + 1) ** 0.5) + (sample / (sigma ** 2 + 1))
        elif self.config.prediction_type == "sample":
            raise NotImplementedError("prediction_type not implemented yet: sample")
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, or `v_prediction`"
            )

        t_, t_next = -self.sigmas[step_index].log(), -self.sigmas[step_index + 1].log()
        h_ = t_next - t_
        if getattr(self, 'pred_ori_sample_prev', None) is None or self.sigmas[step_index + 1] == 0:
            prev_sample = ((-t_next).exp() / (-t_).exp()) * sample - (-h_).expm1() * pred_original_sample
        else:
            h_last = t_ + self.sigmas[step_index - 1].log()
            r_ = h_last / h_
            denoised_d = (1 + 1 / (2 * r_)) * pred_original_sample - (1 / (2 * r_)) * self.pred_ori_sample_prev
            prev_sample = ((-t_next).exp() / (-t_).exp()) * sample - (-h_).expm1() * denoised_d
        self.pred_ori_sample_prev = pred_original_sample

        if not return_dict:
            return (prev_sample,)

        return SchedulerOutput(prev_sample=prev_sample)

    # Copied from diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler.add_noise
    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.FloatTensor,
    ) -> torch.FloatTensor:
        # Make sure sigmas and timesteps have the same device and dtype as original_samples
        sigmas = self.sigmas.to(device=original_samples.device, dtype=original_samples.dtype)
        if original_samples.device.type == "mps" and torch.is_floating_point(timesteps):
            # mps does not support float64
            schedule_timesteps = self.timesteps.to(original_samples.device, dtype=torch.float32)
            timesteps = timesteps.to(original_samples.device, dtype=torch.float32)
        else:
            schedule_timesteps = self.timesteps.to(original_samples.device)
            timesteps = timesteps.to(original_samples.device)

        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)

        noisy_samples = original_samples + noise * sigma
        return noisy_samples

    def __len__(self):
        return self.config.num_train_timesteps

    def get_pred_original_sample(self,
                                 model_output: torch.FloatTensor,
                                 timestep: Union[float, torch.FloatTensor],
                                 sample: torch.FloatTensor):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to(self.timesteps.device)

        step_index = (self.timesteps == timestep).nonzero().item()
        sigma = self.sigmas[step_index]

        # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        if self.config.prediction_type == "epsilon":
            pred_original_sample = sample - sigma * model_output
        elif self.config.prediction_type == "v_prediction":
            # * c_out + input * c_skip
            pred_original_sample = model_output * (-sigma / (sigma ** 2 + 1) ** 0.5) + (sample / (sigma ** 2 + 1))
        elif self.config.prediction_type == "sample":
            raise NotImplementedError("prediction_type not implemented yet: sample")
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, or `v_prediction`"
            )
        return pred_original_sample

    def webui_step(self,
                   pred_original_sample: torch.FloatTensor,
                   timestep: Union[float, torch.FloatTensor],
                   sample: torch.FloatTensor,
                   generator: Optional[torch.Generator] = None):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to(self.timesteps.device)

        step_index = (self.timesteps == timestep).nonzero().item()
        t_, t_next = -self.sigmas[step_index].log(), -self.sigmas[step_index + 1].log()
        h_ = t_next - t_
        if getattr(self, 'pred_ori_sample_prev', None) is None or self.sigmas[step_index + 1] == 0:
            prev_sample = ((-t_next).exp() / (-t_).exp()) * sample - (-h_).expm1() * pred_original_sample
        else:
            h_last = t_ + self.sigmas[step_index - 1].log()
            r_ = h_last / h_
            denoised_d = (1 + 1 / (2 * r_)) * pred_original_sample - (1 / (2 * r_)) * self.pred_ori_sample_prev
            prev_sample = ((-t_next).exp() / (-t_).exp()) * sample - (-h_).expm1() * denoised_d
        self.pred_ori_sample_prev = pred_original_sample

        return prev_sample
