import sys
sys.path.append('./RMBG-1.4')

from briarmbg import BriaRMBG
from model import Wrapper
import skimage.io as io
import torch
import time
import coremltools as ct
from PIL import Image


model = BriaRMBG.from_pretrained("briaai/RMBG-1.4")

wrapper = Wrapper(model)
wrapper.eval()

img_name = "example_input"
image_path = f"./resources/{img_name}.jpg"
orig_im = io.imread(image_path)
orig_im_pt = torch.from_numpy(orig_im)
# make channel first
orig_im_pt = orig_im_pt.permute(2, 0, 1)
orig_im_pt = torch.unsqueeze(orig_im_pt, 0)

traced_model = torch.jit.trace(wrapper, (orig_im_pt,))

coreml_model = ct.convert(
    traced_model,
    inputs = [
        ct.ImageType(name="input", shape=orig_im_pt.shape, color_layout=ct.colorlayout.RGB, )
    ],
)

coreml_model.save("bria-rmbg-coreml.mlpackage")