"""
Glacier/forest segmentation inference script.

Tiles the input image into 512x512 non-overlapping patches,
runs each patch through the model, and reassembles into a
full-resolution binary mask.

Usage:
    python scripts/predict.py \
        --image path/to/image.tif \
        --model checkpoints/attention_unet_glacier_6band.hdf5 \
        --output predictions/mask.png \
        --threshold 0.5
"""

import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from keras.models import load_model

TILE_SIZE = 512


def parse_args():
    parser = argparse.ArgumentParser(description="Run segmentation inference.")
    parser.add_argument("--image",     required=True,       help="Path to input image.")
    parser.add_argument("--model",     required=True,       help="Path to .hdf5 checkpoint.")
    parser.add_argument("--output",    default="mask.png",  help="Output path for predicted mask.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Binarisation threshold (default: 0.5).")
    return parser.parse_args()


def pad_to_tile_multiple(array, tile=TILE_SIZE):
    """Zero-pad height and width to exact multiples of tile size."""
    h, w = array.shape[:2]
    pad_h = (tile - h % tile) % tile
    pad_w = (tile - w % tile) % tile
    return np.pad(array, [[0, pad_h], [0, pad_w], [0, 0]], mode="constant")


def predict(image_path, model_path, output_path, threshold=0.5):
    model = load_model(model_path)

    image = np.array(Image.open(image_path)) / 255.0
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)

    padded = pad_to_tile_multiple(image)
    h_pad, w_pad = padded.shape[:2]
    out_mask = np.zeros((h_pad, w_pad), dtype=np.float32)

    for i in range(0, h_pad, TILE_SIZE):
        for j in range(0, w_pad, TILE_SIZE):
            patch = padded[i:i + TILE_SIZE, j:j + TILE_SIZE, :]
            pred = model.predict(patch[np.newaxis, ...], verbose=0)
            out_mask[i:i + TILE_SIZE, j:j + TILE_SIZE] = pred[0, ..., 0]

    h_orig, w_orig = image.shape[:2]
    final_mask = (out_mask[:h_orig, :w_orig] >= threshold).astype(np.uint8) * 255

    plt.imsave(output_path, final_mask, cmap="gray")
    print(f"Mask saved to: {output_path}")


if __name__ == "__main__":
    args = parse_args()
    predict(args.image, args.model, args.output, args.threshold)