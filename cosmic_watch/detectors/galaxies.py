"""Find and classify sources (stars vs. galaxies) in deep-field images.

v0 is a classical CV pipeline, so it works with no training data:
  1. background model + source extraction (SEP, the Python SExtractor)
  2. per-source morphology: half-light radius and ellipticity
  3. rule-based labels relative to the image's own point-spread function (PSF)

Swap `classify()` for a trained model (e.g. Zoobot or a YOLO head) later; the
rest of the pipeline stays the same.
"""
import cv2
import numpy as np
import sep
from PIL import Image

from ..config import DETECT_MAX_SIDE

Image.MAX_IMAGE_PIXELS = None   # MAST previews can be very large

MAX_SOURCES = 6000
DETECT_SIGMA = 5.0   # threshold in units of background RMS; 3 picks up JPEG noise
MIN_AREA = 15        # pixels above threshold; rejects hot pixels and cosmic rays

COLORS = {   # BGR
    "star": (0, 215, 255),
    "galaxy": (255, 200, 0),
    "elongated galaxy": (255, 0, 200),
    "large galaxy": (0, 255, 0),
}


def _load(path, max_side: int):
    img = Image.open(path)
    scale = min(1.0, max_side / max(img.size))
    if scale < 1.0:
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    rgb = np.asarray(img.convert("RGB"))
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    return rgb, gray, scale


def classify(r50, ellip, psf_r50, saturated=False):
    """Label one source from its shape, relative to the PSF size of this image."""
    # A burnt-out round core is a bright foreground star (its diffraction spikes
    # make it look big, so size alone would call it a galaxy).
    if r50 <= 1.35 * psf_r50 or (saturated and (ellip < 0.4 or ellip > 0.85)):
        return "star"
    if ellip > 0.6:
        return "elongated galaxy"
    if r50 > 4.0 * psf_r50:
        return "large galaxy"
    return "galaxy"


def _footprint_mask(data):
    """Mask pixels outside the survey footprint.

    MAST previews fill the area outside the footprint with flat black (HST) or
    flat white (JWST). Take large flat regions at either extreme that touch the
    image border, and grow them so the noisy footprint edge is masked too.
    Saturated star cores inside the field are kept.
    """
    flat = ((data <= 1) | (data >= 254)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(flat, connectivity=8)
    big = np.zeros_like(flat)
    h, w = data.shape
    min_area = 0.002 * data.size
    for i in range(1, n):
        x, y, bw, bh, area = st[i]
        touches = x == 0 or y == 0 or x + bw >= w or y + bh >= h
        if touches and area >= min_area:
            big[lab == i] = 1
    pad = max(15, round(max(data.shape) / 120))
    return cv2.dilate(big, np.ones((pad, pad), np.uint8)).astype(bool)


def detect(path, max_side: int = DETECT_MAX_SIDE):
    rgb, data, scale = _load(path, max_side)

    nodata = _footprint_mask(data)
    data = np.ascontiguousarray(data)
    bkg = sep.Background(data, mask=nodata, bw=64, bh=64, fw=3, fh=3)
    sub = data - bkg.back()

    sep.set_extract_pixstack(3_000_000)
    sep.set_sub_object_limit(4096)
    objs = sep.extract(sub, DETECT_SIGMA, err=bkg.globalrms, mask=nodata, minarea=MIN_AREA,
                       deblend_cont=0.005, clean=True)
    if len(objs) == 0:
        return rgb, [], {"n_sources": 0, "counts": {}, "scale": scale}

    objs = objs[np.argsort(objs["flux"])[::-1][:MAX_SOURCES]]
    objs = objs[objs["a"] > 0]
    r50, _ = sep.flux_radius(sub, objs["x"], objs["y"], 6.0 * objs["a"], 0.5,
                             normflux=objs["flux"], subpix=5)
    r50 = np.clip(np.nan_to_num(r50, nan=1.0), 0.5, None)
    ellip = 1.0 - objs["b"] / objs["a"]

    # PSF size estimate: the compact end of the bright-source size distribution.
    bright = objs["flux"] > np.percentile(objs["flux"], 50)
    psf_r50 = float(np.percentile(r50[bright], 10)) if bright.sum() > 10 else float(np.percentile(r50, 10))

    h, w = data.shape
    peak_raw = data[np.clip(objs["y"].round().astype(int), 0, h - 1),
                    np.clip(objs["x"].round().astype(int), 0, w - 1)]
    saturated = peak_raw >= 250

    # Diffraction spikes and halos of bright stars break up into fake sources.
    # Drop anything (except other saturated cores) inside a bright star's reach.
    halo = np.zeros(len(objs), bool)
    for i in np.flatnonzero(saturated & (objs["npix"] > 4 * MIN_AREA)):
        reach = min(10.0 * objs["a"][i], 0.25 * max(h, w))
        dist = np.hypot(objs["x"] - objs["x"][i], objs["y"] - objs["y"][i])
        halo |= (dist < reach) & (dist > 0) & ~saturated
    edge = (objs["xmin"] <= 0) | (objs["ymin"] <= 0) | (objs["xmax"] >= w - 1) | (objs["ymax"] >= h - 1)
    keep = ~halo & ~edge
    objs, r50, ellip, saturated = objs[keep], r50[keep], ellip[keep], saturated[keep]

    dets = []
    for o, r, e, s in zip(objs, r50, ellip, saturated):
        label = classify(r, e, psf_r50, bool(s))
        dets.append({
            "label": label,
            "x": float(o["x"]), "y": float(o["y"]),
            "a": float(o["a"]), "b": float(o["b"]), "theta": float(o["theta"]),
            "score": float(o["flux"]),
        })

    counts = {}
    for d in dets:
        counts[d["label"]] = counts.get(d["label"], 0) + 1
    stats = {
        "n_sources": len(dets),
        "n_galaxies": sum(v for k, v in counts.items() if "galaxy" in k),
        "n_stars": counts.get("star", 0),
        "counts": counts,
        "psf_r50_px": round(psf_r50, 2),
        "scale": round(scale, 4),
        "size": [int(rgb.shape[1]), int(rgb.shape[0])],
    }
    return rgb, dets, stats


def annotate(rgb, dets, out_path, max_draw: int = 3000):
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    thick = max(1, round(max(img.shape[:2]) / 1500))
    for d in sorted(dets, key=lambda d: d["score"], reverse=True)[:max_draw]:
        axes = (max(3, int(3 * d["a"])), max(3, int(3 * d["b"])))
        cv2.ellipse(img, (int(d["x"]), int(d["y"])), axes, np.degrees(d["theta"]), 0, 360,
                    COLORS.get(d["label"], (255, 255, 255)), thick, cv2.LINE_AA)
    cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return out_path
