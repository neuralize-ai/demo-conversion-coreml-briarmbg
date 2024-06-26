<div align="center">
    <a href="https://runlocal.ai">
        <img src="./images/large-logo.png" width="300" style="border: none;">
    </a>
</div>
<br>
<div align="center">
    <a href="https://runlocal.ai" style="text-decoration: none;">Website</a> |
    <a href="https://runlocal.ai#contact" style="text-decoration: none;">Contact</a> |
    <a href="https://discord.gg/y9EzZEkwbR" style="text-decoration: none;">Discord</a> |
    <a href="https://x.com/Neuralize_AI" style="text-decoration: none;">Twitter</a>
</div>

<h3 align="center">
   Bria-RMBG-1.4 Conversion Script (CoreML)
</h3>

## :bulb: Introduction
This repo contains a script to convert [Bria-RMBG-1.4](https://huggingface.co/briaai/RMBG-1.4), which is a background removal AI model developed by [Bria AI](https://bria.ai/), from PyTorch to CoreML. After you have converted the model with this script, you can use it in our [Swift Demo App](https://github.com/neuralize-ai/demo-apple-bria-rmbg-swift-app) on an iPhone/iPad.

## :hammer_and_wrench: Setup
Clone this repository:
```
git clone https://github.com/neuralize-ai/demo-conversion-coreml-briarmbg
```

Then, create a conda environment from the `environment.yaml` file:
```
conda env create -f environment.yaml && conda activate demo-conversion-coreml-briarmbg
```


Then, clone 'Bria-RMBG-1.4' from [this Hugging Face repo](https://huggingface.co/briaai/RMBG-1.4):
```
cd demo-apple-bria-rmbg && git clone https://huggingface.co/briaai/RMBG-1.4
```

Then, modify the Hugging Face repo, by replacing `RMBG-1.4/briarmbg.py:5` 
```
from .MyConfig import RMBGConfig
```
with
```
from MyConfig import RMBGConfig
```



 This is needed because the Hugging Face repo expects to be run as its own repo, but we are importing it as a modules, so relative imports cannot be used.

Then, modify `nn.Module` in `RMBG-1.4/briarmbg.py:440`, as below:
```
d1 = self.side1(hx1d)
d1 = _upsample_like(d1,x)

d2 = self.side2(hx2d)
# d2 = _upsample_like(d2,x)

d3 = self.side3(hx3d)
# d3 = _upsample_like(d3,x)

d4 = self.side4(hx4d)
# d4 = _upsample_like(d4,x)

d5 = self.side5(hx5d)
# d5 = _upsample_like(d5,x)

d6 = self.side6(hx6)
# d6 = _upsample_like(d6,x)

# return [F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6)],[hx1d,hx2d,hx3d,hx4d,hx5d,hx6]
return F.sigmoid(d1)
```


## :running: Convert Model 
To run the script with `coremltools`, which is installed in the conda environment, run:
```
python convert.py
```

This will create a `bria-rmbg-coreml.mlpackage` CoreML model file.


## :scroll: License
Bria AI have a custom Hugging Face model license agreement for non-commercial use. Please refer to [their license](https://bria.ai/bria-huggingface-model-license-agreement/) before running this script and/or using the converted model. 
