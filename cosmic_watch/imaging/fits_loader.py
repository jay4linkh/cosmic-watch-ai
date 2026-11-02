"""Read a JWST *_i2d / Hubble *_drc science mosaic: pixels, uncertainty, weights, validity mask and metadata.

These products are already fully calibrated (bias, dark, flat, distortion, drizzle-combined by the STScI pipelines),
so nothing here re-calibrates them; we only read what the pipeline produced:
    JWST i2d : SCI (MJy/sr), ERR (1-sigma), WHT (exposure weight), CON (context)
    HST drc  : SCI (e-/s),   WHT (inverse-variance-like weight), CTX            (no ERR extension)
Pixels outside the footprint are NaN or have zero weight; they are masked, never filled.
"""
import warnings

import numpy as np


def _hdr(h, *keys):
    for k in keys:
        v = h.get(k)
        if v not in (None, "", "N/A", "CLEAR") or (v == "CLEAR" and k == keys[-1]):
            return v
    return None


def load(path):
    from astropy.io import fits
    from astropy.wcs import WCS
    with fits.open(path, memmap=True) as hdul:
        names = {h.name for h in hdul}
        p, sh = hdul[0].header, hdul["SCI"].header
        sci = np.ascontiguousarray(hdul["SCI"].data, dtype=np.float32)
        err = np.ascontiguousarray(hdul["ERR"].data, dtype=np.float32) if "ERR" in names else None
        wht = np.ascontiguousarray(hdul["WHT"].data, dtype=np.float32) if "WHT" in names else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w = WCS(sh)
            try:
                from astropy.wcs.utils import proj_plane_pixel_scales
                pixscale = float(np.mean(proj_plane_pixel_scales(w.celestial))) * 3600
            except Exception:  # noqa: BLE001
                pixscale = None
        filt = _hdr(p, "FILTER", "PUPIL") if p.get("TELESCOP", "").upper() == "JWST" else \
            next((f for f in (p.get("FILTER1"), p.get("FILTER2"), p.get("FILTER")) if f and not str(f).startswith("CLEAR")), None)
        meta = {"telescope": p.get("TELESCOP"), "instrument": p.get("INSTRUME"), "detector": p.get("DETECTOR"),
                "filter": filt, "pupil": p.get("PUPIL"), "exptime": p.get("EFFEXPTM") or p.get("EXPTIME") or p.get("XPOSURE"),
                "bunit": sh.get("BUNIT"), "pixscale_arcsec": pixscale, "shape": list(sci.shape),
                "calibrated": True, "product": p.get("FILENAME") or p.get("ROOTNAME"),
                "cal_version": p.get("CAL_VER") or p.get("CAL_VERS") or p.get("OPUS_VER"),
                "has_err": err is not None, "has_wht": wht is not None}
    valid = np.isfinite(sci)
    if wht is not None:
        valid &= wht > 0
    valid &= sci != 0                         # drizzle leaves exact zeros where nothing landed
    if err is not None:
        err = np.where(np.isfinite(err) & valid, err, np.nan).astype(np.float32)
    sci = np.where(valid, sci, 0).astype(np.float32)
    meta["valid_fraction"] = round(float(valid.mean()), 4)
    return {"sci": sci, "err": err, "wht": wht, "valid": valid, "meta": meta, "header": sh}
