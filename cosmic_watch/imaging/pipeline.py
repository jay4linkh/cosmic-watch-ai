"""Scientific restoration, Phase A: FITS -> background model -> subtraction -> validation -> display stretch.

Every stage is configured by a preset (overridable per call) and recorded in a report, so a render can be
reproduced exactly. Measured pixels stay measured: this phase only removes a smooth background and changes the
display curve, it invents nothing. The FITS file itself is never modified.

    python -m cosmic_watch.imaging.pipeline path/to/file_i2d.fits [preset]
"""
import hashlib
import json
import time

import numpy as np

from . import background, fits_loader, stretching, validation

VERSION = "restore-a2"
PRESETS = {
    # gentle: big mesh, generous source mask, sky keeps its grain
    "conservative": {"mesh": None, "dilate": None, "coarse": True, "stretch": "asinh", "black_sigma": -1.5, "white_pct": 99.95,
                     "beta_sigma": 5.0, "gamma": 1.0, "contrast": 1.0, "brightness": 0.0, "clahe": 0.0},
    # cleaner sky, more contrast in faint structure; validation decides whether it is acceptable
    "aggressive": {"mesh": None, "dilate": None, "coarse": True, "stretch": "asinh", "black_sigma": -1.0, "white_pct": 99.9,
                   "beta_sigma": 2.0, "gamma": 1.0, "contrast": 1.05, "brightness": 0.0, "clahe": 1.5},
}
INSTRUMENT_TWEAKS = {     # adaptive defaults per instrument (Phase A: stretch only; PSF / denoise come later)
    ("JWST", "NIRCAM"): {"beta_sigma_mul": 1.0},
    ("JWST", "MIRI"): {"beta_sigma_mul": 1.5},          # brighter, smoother thermal background
    ("HST", "WFC3"): {"beta_sigma_mul": 1.0},
    ("HST", "ACS"): {"beta_sigma_mul": 1.0},
}


def config(preset="conservative", meta=None, **overrides):
    cfg = dict(PRESETS[preset])
    t = INSTRUMENT_TWEAKS.get(((meta or {}).get("telescope") or "", ((meta or {}).get("instrument") or "").split("/")[0]), {})
    cfg["beta_sigma"] *= t.get("beta_sigma_mul", 1.0)
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


def cache_key(fits_id, preset, cfg):
    blob = json.dumps({"id": fits_id, "v": VERSION, "preset": preset, "cfg": cfg}, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def restore(path, preset="conservative", validate=True, **overrides):
    """Returns (uint8 display image, linear background-subtracted float32, report dict, diagnostics dict)."""
    t, T = time.time(), {}
    d = fits_loader.load(path)
    sci, valid, meta = d["sci"], d["valid"], d["meta"]
    T["load"] = round(time.time() - t, 2)
    cfg = config(preset, meta, **overrides)
    t = time.time()
    bg = background.estimate(sci, valid, mesh=cfg["mesh"], dilate=cfg["dilate"], coarse=cfg["coarse"])
    if cfg.get("mesh_scale") and cfg["mesh"] is None:      # aggressive: smaller boxes follow gradients more closely
        th, cth = bg["thresholds"]
        bg = background.estimate(sci, valid, mesh=max(48, int(bg["mesh"] * cfg["mesh_scale"])), coarse=cfg["coarse"],
                                 thresh=th, coarse_thresh=cth or 2.5, adapt=False)
    lin = (sci - bg["back"]).astype(np.float32)
    lin[~valid] = 0
    T["background"] = round(time.time() - t, 2)
    noise = {"global_rms": bg["global_rms"], "source": "SEP background rms"}
    if d["err"] is not None:
        sky = valid & ~bg["source_mask"] & np.isfinite(d["err"])
        if sky.any():
            noise["pipeline_err_median"] = float(np.median(d["err"][sky]))
            noise["rms_over_err"] = round(bg["global_rms"] / max(noise["pipeline_err_median"], 1e-12), 3)
    report = {"version": VERSION, "preset": preset, "config": cfg, "meta": meta,
              "background": {k: bg.get(k) for k in ("mesh", "dilate", "filled_boxes", "thresholds", "global_back", "global_rms", "masked_fraction", "sky_pixels")},
              "noise": noise,
              "uniformity": {"before": background.uniformity((sci - np.median(sci[valid])).astype(np.float32), valid, bg["source_mask"]),
                             "after": background.uniformity(lin, valid, bg["source_mask"])},
              "provenance": {"pixels": "measured", "operations": ["smooth background subtracted (SEP, sources masked)", f"{cfg['stretch']} display stretch"],
                             "invented_pixels": False, "fits_modified": False}}
    if validate:
        t = time.time()
        report["validation"] = validation.compare(sci, lin, valid, bg["rms"], bg["global_rms"], sky_fraction=1 - bg["masked_fraction"])
        mesh_used = bg["mesh"]
        th, cth = bg["thresholds"]
        est = lambda img, v: background.estimate(img, v, mesh=mesh_used, dilate=cfg["dilate"], coarse=cfg["coarse"],  # noqa: E731
                                                 thresh=th, coarse_thresh=cth or 2.5, adapt=False)
        report["injection"] = validation.injection_test(sci, valid, lin, bg["source_mask"], bg["global_rms"], est)
        report["validation"]["passed"] = report["validation"]["passed"] and report["injection"].get("passed", True)
        T["validation"] = round(time.time() - t, 2)
    t = time.time()
    img = stretching.stretch(lin, valid, bg["global_rms"], method=cfg["stretch"], black_sigma=cfg["black_sigma"],
                             white_pct=cfg["white_pct"], beta_sigma=cfg["beta_sigma"], gamma=cfg["gamma"],
                             contrast=cfg["contrast"], brightness=cfg["brightness"], clahe=cfg["clahe"])
    T["stretch"] = round(time.time() - t, 2)
    report["timings_s"] = T
    diag = {"background": bg["back"], "rms": bg["rms"], "source_mask": bg["source_mask"], "valid": valid}
    return img, lin, report, diag


def diagnostic_png(diag, path, max_side=1024):
    """Background map | source mask side by side, downsampled (for the viewer and debugging)."""
    import cv2
    b, m, v = diag["background"], diag["source_mask"], diag["valid"]
    k = max(1, max(b.shape) // max_side)
    b, m, v = b[::k, ::k], m[::k, ::k], v[::k, ::k]
    lo, hi = np.percentile(b[v], [1, 99]) if v.any() else (0, 1)
    bb = (np.clip((b - lo) / max(hi - lo, 1e-12), 0, 1) * 255).astype(np.uint8)
    bb = cv2.applyColorMap(bb, cv2.COLORMAP_VIRIDIS)
    mm = np.dstack([m * 255] * 3).astype(np.uint8)
    bb[~v] = 0
    cv2.imwrite(str(path), np.hstack([bb, mm]))


if __name__ == "__main__":
    import sys
    img, _, rep, _ = restore(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "conservative")
    print(json.dumps({k: v for k, v in rep.items() if k != "config"}, indent=1, default=str))
