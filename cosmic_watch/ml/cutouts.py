"""Shared preprocessing: cut a galaxy out of an image exactly the same way for training and inference."""
import cv2
import numpy as np

SIZE = 128                     # network input (px)
MEAN, STD = 0.449, 0.226       # ImageNet grey statistics (backbone was pretrained on ImageNet)
FIELD = 6.0                    # cut-out half-width in units of the source's semi-major axis


def cutout(gray: np.ndarray, x: float, y: float, a: float, min_half: int = 20) -> np.ndarray:
    """Square cut-out centred on (x, y) in full-resolution pixels, resized to SIZE x SIZE.

    Galaxy10 cut-outs show the galaxy filling roughly the middle half of the frame; a
    half-width of FIELD * a (SEP semi-major axis) matches that framing.
    """
    h, w = gray.shape
    half = int(max(min_half, FIELD * a))
    pad = cv2.copyMakeBorder(gray, half, half, half, half, cv2.BORDER_CONSTANT, value=0)
    cx, cy = int(round(x)) + half, int(round(y)) + half
    crop = pad[cy - half:cy + half, cx - half:cx + half]
    return cv2.resize(crop, (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def normalize(batch_u8: np.ndarray) -> np.ndarray:
    """uint8 (N, H, W) -> float32 (N, 3, H, W) normalised, grey replicated to 3 channels."""
    x = batch_u8.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return np.repeat(x[:, None], 3, axis=1)
