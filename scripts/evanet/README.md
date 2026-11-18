# EvaNet (vendored)

The model and loss code in this folder is copied from the EvaNet repository:

- **Source:** https://github.com/MTSami/EvaNet
- **Commit:** `763544cd6e31280371b98d827d3913bcfd5552ce`
- **Folder copied from:** `Eva_Net_4_channel/`
- **License:** MIT, Copyright (c) 2022 Mirza Tanzim Sami (see `LICENSE`)

## Files

| File | What it is |
|---|---|
| `elev_conv.py` | `ElevationConv`: image conv whose output is gated by `sigmoid(conv(elevation))` |
| `elev_deconv.py` | `ElevationConvTranspose`: the same gating for upsampling |
| `eva_net_parts.py` | U-Net blocks (`double_conv`, `down`, `up`, ...) built from the two layers above |
| `eva_net_model.py` | `EvaNet`, the full encoder-decoder |
| `loss.py` | `ElevationLoss`, the gravity loss over each pixel's 3x3 neighbourhood |

`gated_resnet.py` is **not from upstream**. It's this project's hybrid: EvaNet's elevation-gated
convolutions on an ImageNet-pretrained ResNet50 U-Net laid out like `notebooks/unet_dem.ipynb`
(trained in `notebooks/evanet_resnet.ipynb`).

Their `main.py`, data loading, config and checkpoints were not copied. Training and data loading
live in `notebooks/evanet.ipynb`, which uses this project's own tiles from `scripts/dataset.py`.

## Changes from upstream

- Flat imports (`from elev_conv import *`) changed to relative imports (`from .elev_conv import *`)
  so the folder works as a package: `from scripts.evanet import EvaNet, ElevationLoss`.
- Added `__init__.py` and this README.
- **Bug fix in `loss.py`:** `torch.argmax(pred_labels, dim=1)` became
  `torch.argmax(pred_labels, dim=1, keepdim=True)`. Without `keepdim`, the prediction mask is
  (B, H, W). It broadcasts against the (B, 1, H, W) labels to (B, B, H, W), so each sample's
  prediction is also scored against every other sample's mask in the batch. With batch size 3 on
  random data, upstream gave 7463.8 versus 2488.1 for the sum of per-sample losses. After the fix
  the two match. Upstream trains with batch size 4, so their published runs include this effect.

Nothing else was changed.

## Usage notes

- Use `EvaNet(batch_size, n_channels=3, n_classes=2, ultrasmall=True)`. The `ultrasmall=False`
  branch passes the wrong arguments to `down()`/`up()` and raises an error.
- Output channel **0 is flood** and channel 1 is dry, so the flood mask is `pred.argmax(1) == 0`.
- `ElevationConv` creates its conv layers on CUDA when a GPU is available, so the model has to stay on the GPU.
- `ElevationLoss(pred, elev, labels)` expects labels as flood = +1, dry = -1, unknown = 0
  (this project's tiles use 1 / 0 / -1, so they need converting). It returns a **sum** over pixels,
  which is why upstream trains with SGD at lr = 1e-6.
