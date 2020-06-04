# GANimation_TL

A tensorlayer implementation of GANimation

paper: https://arxiv.org/abs/1807.09251

Author's implementation: https://github.com/albertpumarola/GANimation

Reference: https://github.com/hao-qiang/GANimation-tf

## Requirements
- tensorflow-gpu 2.0.0ac
- tensorlayer 2.2.2

## Data

Reference: https://github.com/hao-qiang/GANimation-tf

## Note

In the 'model' module implemented with tensorlayer2.2.2, during training, the InstanceNorm2d layer would have problems with the error 'incompatible shapes [1,1,1,1600] vs. [1,1,1,64]' (the input is [25,128,128,20](B,H,W,C)). The reason was not found, so we changed to version 1.11.0 later.

## Model

Re-implement model in TensorLayer

## Training
```
python train.py
```

## Testing
```
python test.py
```
