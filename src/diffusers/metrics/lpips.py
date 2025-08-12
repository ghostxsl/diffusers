
import torch
import lpips


__all__ = ['LPIPSMetric']


class LPIPSMetric(object):
    def __init__(self, net='vgg', device=torch.device("cuda")):
        self.net = net
        self.device = device
        self.model = lpips.LPIPS(net=net).to(self.device)

    @torch.no_grad()
    def __call__(self, image_1, image_2, normalize=True):
        """
            image_1: images with size (n, 3, w, h) with value [-1, 1]
            image_2: images with size (n, 3, w, h) with value [-1, 1]
        """
        image_1 = image_1.to(self.device)
        image_2 = image_2.to(self.device)
        result = self.model.forward(image_1, image_2, normalize=normalize)
        return result
