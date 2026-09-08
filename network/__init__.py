"""Network package for LSFSeg.

Preferred module names:
- data: MGU dataset loading and augmentation.
- models: label encoder/decoder, image encoder, and MeanFlow denoiser.
- training: Stage1/Stage2 training loops and losses.
- inference: testing, sampling, metrics, and visualization.
- fourier: Fast Fourier Convolution blocks.
- utils: shared tensor, noise, plotting, and device helpers.

Legacy module names are kept as aliases so older scripts can still import
network.model_pytorch, network.dataLoader_pytorch, and related names.
"""

import sys

from . import data, fourier, inference, models, training, utils

dataLoader_pytorch = data
model_pytorch = models
sampler_pytorch = inference
trainer_pytorch = training
misc_pytorch = utils
ffc = fourier

sys.modules[f"{__name__}.dataLoader_pytorch"] = data
sys.modules[f"{__name__}.model_pytorch"] = models
sys.modules[f"{__name__}.sampler_pytorch"] = inference
sys.modules[f"{__name__}.trainer_pytorch"] = training
sys.modules[f"{__name__}.misc_pytorch"] = utils
sys.modules[f"{__name__}.ffc"] = fourier
