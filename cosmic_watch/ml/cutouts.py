"""Shared preprocessing: cut a galaxy out of an image exactly the same way for training and inference."""
import cv2
import numpy as np

SIZE = 128                     # v1 network input (px); each model card records its own size
MEAN, STD = 0.449, 0.226       # ImageNet grey statistics (v1 backbone was pretrained on ImageNet)
FIELD = 6.0                    # cut-out half-width in units of the source's semi-major axis


def cutout(gray: np.ndarray, x: float, y: float, a: float, min_half: int = 20, size: int = SIZE) -> np.ndarray:
    """Square cut-out centred on (x, y) in full-resolution pixels, resized to size x size.

    Galaxy10 cut-outs show the galaxy filling roughly the middle half of the frame; a
    half-width of FIELD * a (SEP semi-major axis) matches that framing.
    """
    h, w = gray.shape
    half = int(max(min_half, FIELD * a))
    pad = cv2.copyMakeBorder(gray, half, half, half, half, cv2.BORDER_CONSTANT, value=0)
    cx, cy = int(round(x)) + half, int(round(y)) + half
    crop = pad[cy - half:cy + half, cx - half:cx + half]
    interp = cv2.INTER_AREA if 2 * half >= size else cv2.INTER_CUBIC
    return cv2.resize(crop, (size, size), interpolation=interp)


def normalize(batch_u8: np.ndarray, mean: float = MEAN, std: float = STD, channels: int = 3) -> np.ndarray:
    """uint8 (N, H, W) -> float32 (N, C, H, W) normalised (grey replicated when C = 3)."""
    x = (batch_u8.astype(np.float32) / 255.0 - mean) / std
    return np.repeat(x[:, None], channels, axis=1)
