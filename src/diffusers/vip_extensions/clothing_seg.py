# Copyright (c) wilson.xu. All rights reserved.
from PIL import Image
import torch
from torchvision import transforms


__all__ = ['ClothSeg']


class ClothSeg(object):
    def __init__(self, model_path="cloth_seg_v5.0.pt", input_size=1024, device="cpu"):
        self.cloth_model = torch.jit.load(model_path)
        self.cloth_model.eval()
        self.cloth_model = self.cloth_model.to(device)
        self.device = device

        self.cloth_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    def __call__(self, image):
        with torch.no_grad():
            input = self.cloth_transform(image).unsqueeze(0).to(self.device)
            mask_pred = self.cloth_model(input)[0]
            mask_pred = torch.argmax(mask_pred, dim=1)

        mask_cloth = mask_pred[0].cpu().numpy().astype(np.uint8)
        mask_cloth[mask_cloth != 0] = 255

        return Image.fromarray(mask_cloth)
