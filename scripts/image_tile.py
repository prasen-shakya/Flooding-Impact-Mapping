import rasterio
from rasterio.windows import Window
import os

def tile_image(img_path, mask_path, out_dir, prefix, tile=256, stride=128):
    os.makedirs(f"{out_dir}/images", exist_ok=True)
    os.makedirs(f"{out_dir}/masks", exist_ok=True)

    with rasterio.open(img_path) as src, rasterio.open(mask_path) as msk:
        H, W = src.height, src.width
        n = 0
        for y in range(0, H - tile + 1, stride):
            for x in range(0, W - tile + 1, stride):
                win = Window(x, y, tile, tile)
                img = src.read(window=win)
                label = msk.read(1, window=win)

                transform = src.window_transform(win)

                img_profile = src.profile | {"width": tile, "height": tile, "transform": transform}
                with rasterio.open(f"{out_dir}/images/{prefix}_{n}.tif", "w", **img_profile) as dst:
                    dst.write(img)

                mask_profile = msk.profile | {"width": tile, "height": tile, "transform": transform}
                with rasterio.open(f"{out_dir}/masks/{prefix}_{n}.tif", "w", **mask_profile) as dst:
                    dst.write(label, 1)
                n += 1
    return n


TRAIN_SCENES = range(4, 13)
TEST_SCENES = range(1, 4)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    untiled_data_path = os.path.normpath(os.path.join(base_dir, "..", "data", "untiled_data"))
    tiled_data_path = os.path.normpath(os.path.join(base_dir, "..", "data", "tiled_data"))

    splits = {"train": TRAIN_SCENES, "test": TEST_SCENES}
    for split, scenes in splits.items():
        total = 0
        for i in scenes:
            scene = f"image_{i}"
            scene_dir = os.path.join(untiled_data_path, scene)

            n = tile_image(
                img_path=os.path.join(scene_dir, "image.tif"),
                mask_path=os.path.join(scene_dir, "label.tif"),
                out_dir=os.path.join(tiled_data_path, split),
                prefix=scene,
            )
            print(f"{split}/{scene}: {n} tiles")
            total += n
        print(f"{split} total: {total} tiles")


if __name__ == "__main__":
    main()
