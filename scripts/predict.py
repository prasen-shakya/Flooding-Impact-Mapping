"""Runs a trained U-Net over whole images and saves the flood probability map as a GeoTIFF.

Works with the weights from any of the notebooks. The number of input channels is read from the weights file:
- 3 = RGB only (unet_pretrained, model_using_floodnet, julian_unet_rgb_*): no extra options
- 4 = RGB + one DEM channel: say how that DEM was prepared with --dem, because it changed over time
    --dem metres  DEM minus the image mean, x20  (dataset.DEM_STANDARDIZE = False; julian_* folds, weights before 2026-11-10)
    --dem scaled  DEM standardized per image, x40 (dataset.DEM_STANDARDIZE = True; current unet_dem.ipynb)
- 5 = RGB + both DEM channels (unet_dem_2dem, unet_dem_eva_2dem): metres x20 and scaled x40, picked automatically
Getting --dem wrong doesn't crash, it just gives bad predictions, so it has no default.

Each image is cut into 256x256 tiles with half overlap (stride 128). Where tiles overlap, their
predictions are averaged, which hides the seams at tile edges.

Output: data/image_<i>/pred_<weights name>.tif, float32 flood probability 0-1, same grid as image.tif.
Pixels with no image data are -1.

Run from anywhere:
    python scripts/predict.py outputs/unet_rgb_fold_nc_seed1.weights.h5            # all 25 images
    python scripts/predict.py outputs/unet_dem.weights.h5 --dem scaled --images 4 5 6
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
import rasterio

import dataset
from dataset import DATA_DIR, TILE, load_image, make_input

NUM_IMAGES = 25
# How each DEM mode was prepared in the notebooks: the dataset.DEM_STANDARDIZE setting, then the multiplier
DEM_MODES = {
    "metres": (False, 20),
    "scaled": (True, 40),
    "both": ("both", np.float32([20, 40])),
}


def build_unet(channels):
    """Same ResNet50 U-Net as the notebooks. No ImageNet download: the saved weights replace everything."""
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.layers import (Activation, BatchNormalization, Concatenate, Conv2D,
                                         Conv2DTranspose, Input)
    from tensorflow.keras.models import Model

    def conv_block(x, n):
        for _ in range(2):
            x = Conv2D(n, 3, padding="same")(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)
        return x

    def decoder_block(x, skip, n):
        x = Conv2DTranspose(n, 2, strides=2, padding="same")(x)
        return conv_block(Concatenate()([x, skip]), n)

    inputs = Input((TILE, TILE, channels))
    resnet50 = ResNet50(include_top=False, weights=None, input_tensor=inputs)
    s2 = resnet50.get_layer("conv1_relu").output
    s3 = resnet50.get_layer("conv2_block3_out").output
    s4 = resnet50.get_layer("conv3_block4_out").output
    b1 = resnet50.get_layer("conv4_block6_out").output
    d = decoder_block(b1, s4, 512)
    d = decoder_block(d, s3, 256)
    d = decoder_block(d, s2, 128)
    d = decoder_block(d, inputs, 64)
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d)
    return Model(inputs, outputs)


def weights_channels(path):
    """3 or 4: reads the input depth of conv1_conv's kernel (shape 7x7xCx64) from the weights file."""
    found = []
    with h5py.File(path, "r") as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset) and obj.ndim == 4 and obj.shape[:2] == (7, 7):
                found.append(obj.shape[2])
        f.visititems(visit)
    if not found:
        raise ValueError(f"Couldn't find conv1_conv's 7x7 kernel in {path}")
    return found[0]


def load_model(path):
    channels = weights_channels(path)
    model = build_unet(channels)
    model.load_weights(path)
    return model, channels


def dem_mode(channels, dem):
    """Checks --dem against the model's channel count. Returns the DEM mode, or None for RGB models."""
    if channels == 3:
        return None
    if channels == 5:
        return "both"
    if channels == 4:
        if dem not in ("metres", "scaled"):
            raise SystemExit("This model has a DEM channel: add --dem metres or --dem scaled (see the top of predict.py)")
        return dem
    raise SystemExit(f"Don't know how to prepare {channels} input channels")


def preprocess(x, mode):
    """Same as the notebooks: ResNet50 preprocessing on RGB, DEM channel(s) times their multiplier."""
    from tensorflow.keras.applications.resnet50 import preprocess_input
    rgb = preprocess_input(x[..., :3] * 255.0)
    if mode is None:
        return rgb.astype(np.float32)
    return np.concatenate([rgb, x[..., 3:] * DEM_MODES[mode][1]], axis=-1).astype(np.float32)


def predict_image(model, mode, i, stride=TILE // 2, batch=8):
    """Flood probability for every pixel of image i, shape (H, W). No-data pixels are -1.
    mode: None (RGB model), "metres", "scaled" or "both" (see dem_mode)."""
    rgb, dem, label = load_image(i)  # label is only used for its no-data mask
    dataset.DEM_STANDARDIZE = DEM_MODES[mode][0] if mode else False  # RGB models ignore the DEM
    x = make_input(rgb, dem)
    H, W = label.shape
    rows = list(range(0, H - TILE, stride)) + [H - TILE]
    cols = list(range(0, W - TILE, stride)) + [W - TILE]
    corners = [(r, c) for r in rows for c in cols]

    total = np.zeros((H, W), np.float32)
    count = np.zeros((H, W), np.float32)
    for k in range(0, len(corners), batch):
        chunk = corners[k:k + batch]
        tiles = np.stack([x[r:r + TILE, c:c + TILE] for r, c in chunk])
        probs = model.predict(preprocess(tiles, mode), verbose=0)[..., 0]
        for (r, c), p in zip(chunk, probs):
            total[r:r + TILE, c:c + TILE] += p
            count[r:r + TILE, c:c + TILE] += 1

    prob = total / count
    with rasterio.open(DATA_DIR / f"image_{i}" / "image.tif") as src:
        alpha = src.read(4)
    prob[alpha == 0] = -1
    return prob


def pred_path(i, name):
    return DATA_DIR / f"image_{i}" / f"pred_{name}.tif"


def model_name(weights_path):
    """unet_dem.weights.h5 -> unet_dem"""
    return Path(weights_path).name.split(".")[0]


def save_prediction(prob, i, name):
    with rasterio.open(DATA_DIR / f"image_{i}" / "image.tif") as src:
        profile = src.profile
    profile.update(count=1, dtype="float32", nodata=-1, compress="deflate")
    with rasterio.open(pred_path(i, name), "w", **profile) as dst:
        dst.write(prob, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("weights", help="a .weights.h5 file saved by one of the notebooks")
    parser.add_argument("--images", type=int, nargs="+", default=range(1, NUM_IMAGES + 1))
    parser.add_argument("--dem", choices=["metres", "scaled"], help="how the DEM was prepared (4-channel models only)")
    args = parser.parse_args()

    channels = weights_channels(args.weights)
    mode = dem_mode(channels, args.dem)  # check before the slow model build
    model, _ = load_model(args.weights)
    name = model_name(args.weights)
    print(f"{name}: {channels} input channels, DEM: {mode or 'none'}")
    for i in args.images:
        if not (DATA_DIR / f"image_{i}" / "image.tif").exists():
            print(f"image_{i}: no image.tif, skipping")
            continue
        prob = predict_image(model, mode, i)
        save_prediction(prob, i, name)
        print(f"image_{i}: wrote {pred_path(i, name).name}")


if __name__ == "__main__":
    main()
