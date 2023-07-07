import os
import torch
import torch.nn as nn
import torch.nn.init as init
from safetensors.torch import load_file, save_file

from diffusers.utils import logging, SAFETENSORS_WEIGHTS_NAME, WEIGHTS_NAME


logger = logging.get_logger(__name__)


class PoseGuider(nn.Module):
    def __init__(self, noise_latent_channels=4):
        super(PoseGuider, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Final projection layer
        self.final_proj = nn.Conv2d(in_channels=128, out_channels=noise_latent_channels, kernel_size=1)

        # Initialize layers
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize weights with Gaussian distribution and zero out the final layer
        for m in self.conv_layers:
            if isinstance(m, nn.Conv2d):
                init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    init.zeros_(m.bias)

        init.zeros_(self.final_proj.weight)
        if self.final_proj.bias is not None:
            init.zeros_(self.final_proj.bias)

    def forward(self, pose_image):
        x = self.conv_layers(pose_image)
        x = self.final_proj(x)

        return x

    @classmethod
    def from_pretrained(cls,
                        pretrained_model_path,
                        subfolder=None,
                        use_safetensors=True,
                        **kwargs):
        if not os.path.exists(pretrained_model_path):
            print(f"There is no model file in {pretrained_model_path}")
        if subfolder is not None:
            pretrained_model_path = os.path.join(pretrained_model_path, subfolder)
        print(f"loaded PoseGuider's pretrained weights from {pretrained_model_path}...")

        weights_name = SAFETENSORS_WEIGHTS_NAME if use_safetensors else WEIGHTS_NAME
        weights_name = kwargs.get("weights_name", None) or weights_name

        if use_safetensors:
            state_dict = load_file(os.path.join(pretrained_model_path, weights_name))
        else:
            state_dict = torch.load(os.path.join(pretrained_model_path, weights_name),
                                    map_location="cpu")

        print("### Initial model: PoseGuider...")
        model = cls()
        m, u = model.load_state_dict(state_dict, strict=False)
        print(f"### missing keys: {len(m)}; \n### unexpected keys: {len(u)};")

        return model

    def save_pretrained(self,
                        save_directory,
                        safe_serialization=True,
                        **kwargs):
        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        os.makedirs(save_directory, exist_ok=True)

        # Save the model
        state_dict = self.state_dict()

        weights_name = SAFETENSORS_WEIGHTS_NAME if safe_serialization else WEIGHTS_NAME
        weights_name = kwargs.get("weights_name", None) or weights_name

        # Save the model
        if safe_serialization:
            save_file(
                state_dict, os.path.join(save_directory, weights_name), metadata={"format": "pt"}
            )
        else:
            torch.save(state_dict, os.path.join(save_directory, weights_name))

        logger.info(f"Model weights saved in {os.path.join(save_directory, weights_name)}")
