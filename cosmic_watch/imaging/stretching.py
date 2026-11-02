"""Display stretches for background-subtracted linear data (display only: never used for measurement).

Black point sits slightly BELOW the sky (default -1 sigma) so the sky keeps a faint natural grain instead of being
crushed to pure black; white point is a high percentile of the valid pixels so cores keep some structure.
Methods: asinh (default, Lupton-style softening in units of the noise), log, sqrt, linear, zscale, percentile,
histogram equalisation; optional CLAHE on top for local contrast.
"""
import cv2
import numpy as np

METHODS = ("asinh", "log", "sqrt", "linear", "zscale", "percentile", "histeq")


def limits(lin, valid, rms, black_sigma=-1.0, white_pct=99.95, method="asinh"):
    vals = lin[valid]
    if vals.size > 4_000_000:
        vals = vals[:: vals.size // 4_000_000]
    if method == "zscale":
        from astropy.visualization import ZScaleInterval
        return tuple(float(v) for v in ZScaleInterval().get_limits(vals))
    if method == "percentile":
        lo, hi = np.percentile(vals, [1, white_pct])
        return float(lo), float(hi)
    black = black_sigma * rms
    white = float(np.percentile(vals, white_pct))
    return float(black), max(white, black + 10 * rms)


def stretch(lin, valid, rms, method="asinh", black_sigma=-1.0, white_pct=99.95, beta_sigma=2.0, gamma=1.0,
            contrast=1.0, brightness=0.0, clahe=0.0):
    """lin: background-subtracted float32. rms: global sky noise (float). Returns uint8."""
    if method not in METHODS:
        raise ValueError(f"unknown stretch {method}")
    lo, hi = limits(lin, valid, rms, black_sigma, white_pct, method)
    x = (lin - lo) / max(hi - lo, 1e-12)
    if method == "asinh":
        beta = max(beta_sigma * rms, 1e-12) / max(hi - lo, 1e-12)
        y = np.arcsinh(np.clip(x, 0, None) / beta) / np.arcsinh(1 / beta)
    elif method == "log":
        a = 1000.0
        y = np.log1p(a * np.clip(x, 0, None)) / np.log1p(a)
    elif method == "sqrt":
        y = np.sqrt(np.clip(x, 0, None))
    elif method == "histeq":
        y8 = (np.clip(x, 0, 1) * 255).astype(np.uint8)
        y = cv2.equalizeHist(y8).astype(np.float32) / 255
    else:
        y = x
    y = np.clip(y, 0, 1)
    if gamma != 1.0:
        y = y ** (1 / gamma)
    if contrast != 1.0 or brightness:
        y = np.clip((y - 0.5) * contrast + 0.5 + brightness, 0, 1)
    out = (y * 255 + 0.5).astype(np.uint8)
    if clahe > 0:
        out = cv2.createCLAHE(clipLimit=float(clahe), tileGridSize=(16, 16)).apply(out)
    out[~valid] = 0
    return out
