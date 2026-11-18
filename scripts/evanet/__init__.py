"""EvaNet model and gravity loss, vendored from https://github.com/MTSami/EvaNet (MIT license).

See README.md in this folder for the source commit and what was changed.
"""
from .eva_net_model import EvaNet
from .loss import ElevationLoss
from .gated_resnet import GatedResNetUNet, IMAGENET_MEAN, IMAGENET_STD  # this project's hybrid, not upstream

__all__ = ["EvaNet", "ElevationLoss", "GatedResNetUNet", "IMAGENET_MEAN", "IMAGENET_STD"]
