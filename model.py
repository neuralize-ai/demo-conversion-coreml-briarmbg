import torch
from PIL import Image
import torch.nn.functional as F
from torch import nn
from typing import Optional, List
from torchvision.transforms.functional import normalize


class Wrapper(nn.Module):
    def __init__(self, model):
        super(Wrapper, self).__init__()
        self.model = model

    def preprocess_image(self, im_tensor: torch.Tensor, model_input_size: Optional[List[int]]) -> torch.Tensor:
        im_tensor = F.interpolate(im_tensor, size=model_input_size, mode='bilinear')
        image = torch.divide(im_tensor, 255.0)
        image = normalize(image, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
        return image

    def postprocess_image(self, result: torch.Tensor, im_size: Optional[List[int]]) -> torch.Tensor:
        result = torch.squeeze(F.interpolate(result, size=im_size, mode='bilinear'), 0)

        ma = torch.max(result)
        mi = torch.min(result)
        result = (result - mi) / (ma - mi)
        im_array = (result * 255)
        im_array = torch.unsqueeze(im_array, 0)
        # im_array = torch.squeeze(im_array, dim=2)
        return im_array


    def forward(self, orig_im_pt: torch.Tensor):
        model_input_size = [1024, 1024]
        # im = self.preprocess_image(orig_im_pt, model_input_size)
        result = self.model(orig_im_pt)
        result_image = self.postprocess_image(result, model_input_size)
        return result_image