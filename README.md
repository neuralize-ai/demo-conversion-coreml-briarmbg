## CoreML Conversion Script For BriaRMBG Background Removal

### First, clone this directory
```
git clone https://github.com/neuralize-ai/demo-apple-bria-rmbg.git
```

Next, create a conda environment from the `environment.yaml` file
```
conda env create -f environment.yaml && conda activate demo-apple-bria-rmbg-conversion
```


### Next, clone the 'RMBG-1.4'model repo from Hugging Face into this repo
```
cd demo-apple-bria-rmbg && git clone https://huggingface.co/briaai/RMBG-1.4
```


### Now we need to slightly modify the model repo

Inside
```
RMBG-1.4/briarmbg.py:5
```

Comment out and add this line. We need to do this because the model repo is written to expect it to be run as its own repo, but we are importing as one of our modules, so relative imports cannot be used
```
# from .MyConfig import RMBGConfig
from MyConfig import RMBGConfig
```

Now, modify the `nn.Module` because it contains many outputs that we do not use

In `RMBG-1.4/briarmbg.py:440`, comment out some lines as shown below
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


### Convert with `coremltools`
Now we can do `python convert.py`

This will create a CoreML file at the source root called `bria-rmbg-coreml.mlpackage`


### Disclaimers
This model is traced using the dimensions of the example image, which means only images with the same dimensions can be used. The right way to do this is to only trace the `forward` method, and not the `preprocess` and `postprocess` models. But this is just for demo purposes.
