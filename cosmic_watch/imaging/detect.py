"""Science-grade source detection on the calibrated FITS mosaic, at several scales.

The old detector ran one threshold on the 8-bit JPEG preview. This one works on the float data with its noise map:

    cold   broad matched filter, little deblending   -> bright and extended galaxies, kept whole
    hot    PSF-sized matched filter, strong deblend  -> faint and tiny sources the cold pass misses
    lsb    4x and 8x binned, after masking the above -> large diffuse (low-surface-brightness) galaxies

Each pass is run on the sign-flipped image too: whatever it "finds" there is pure noise, which gives the spurious
rate. Thresholds are raised until that rate is under MAX_SPURIOUS, so the catalogue's purity is measured, not assumed.
Sources get Kron fluxes, AB magnitudes (from the header's photometric keywords), half-light radii, sky positions and a
size class. Completeness is measured by injecting fake galaxies of known size and brightness (see `completeness`).

    python -m cosmic_watch.imaging.detect <image_id>
"""
import logging
import time
import warnings

import numpy as np
import sep

from . import background, fits_loader

log = logging.getLogger(__name__)
VERSION = "sci-det-1"
MAX_SPURIOUS = 0.03
MODES = {     # threshold (sigma of the filtered image), min area px, kernel FWHM in PSF FWHMs, deblend contrast, binning
    "cold": {"thresh": 3.5, "minarea": 12, "kernel": 2.0, "deblend": 0.02, "bin": 1},
    "hot": {"thresh": 4.0, "minarea": 4, "kernel": 1.0, "deblend": 0.005, "bin": 1},
    "lsb4": {"thresh": 3.5, "minarea": 6, "kernel": 1.5, "deblend": 0.05, "bin": 4},
    "lsb8": {"thresh": 3.5, "minarea": 6, "kernel": 1.5, "deblend": 0.05, "bin": 8},
}
SIZE_CLASSES = (("micro", 2.0), ("small", 4.0), ("medium", 8.0), ("large", 16.0), ("giant", np.inf))   # r50 / PSF r50 upper bounds
STAR_MAX = 1.35                    # r50 <= this many PSF r50 = unresolved

sep.set_extract_pixstack(20_000_000)
sep.set_sub_object_limit(8192)


def _kernel(fwhm):
    fwhm = max(1.5, float(fwhm))
    s = fwhm / 2.355
    n = int(np.ceil(3 * s)) * 2 + 1
    y, x = np.mgrid[:n, :n] - n // 2
    k = np.exp(-(x * x + y * y) / (2 * s * s))
    return (k / k.sum()).astype(np.float32)


def _bin(a, k, fn=np.mean):
    h, w = (a.shape[0] // k) * k, (a.shape[1] // k) * k
    return np.ascontiguousarray(fn(a[:h, :w].reshape(h // k, k, w // k, k), axis=(1, 3)), dtype=np.float32)


def _detimg(img, rms, mask, sky, fwhm):
    """Detection image in true sigma units: whitened, smoothed with the kernel, then scaled by the noise measured on
    empty sky. Drizzled pixels are correlated, so the textbook matched-filter normalisation is too optimistic."""
    import cv2
    w = np.where(mask, 0, img / rms).astype(np.float32)
    d = cv2.filter2D(w, -1, _kernel(fwhm), borderType=cv2.BORDER_REFLECT)
    s = d[sky & ~mask]
    if s.size < 500:
        s = d[~mask]
    med = float(np.median(s))
    sig = float(1.4826 * np.median(np.abs(s - med))) or 1.0
    return np.ascontiguousarray((d - med) / sig, dtype=np.float32)


def _extract(D, mask, m, thresh, fast=False):
    """Sources in a detection image that is already in sigma units. fast = no deblending: enough for counting noise
    peaks and for finding isolated injected sources."""
    kw = dict(mask=mask, minarea=m["minarea"], filter_kernel=None, clean=True, segmentation_map=True)
    try:
        return sep.extract(D, thresh, deblend_nthresh=16, deblend_cont=1.0 if fast else m["deblend"], **kw)
    except Exception as e:  # noqa: BLE001  (deblending overflow in a pathological field: retry without deblending)
        log.warning("extract retry without deblending: %s", e)
        return sep.extract(D, thresh, deblend_cont=1.0, **kw)


def psf_r50(lin, rms, mask):
    """Half-light radius of point sources: the compact end of the bright, unsaturated sources."""
    o = sep.extract(lin, 20.0, err=rms, mask=mask, minarea=6, deblend_cont=1.0)
    o = o[(o["a"] > 0) & (o["flag"] < 4)]
    if len(o) < 8:
        return 1.5
    r, _ = sep.flux_radius(lin, o["x"], o["y"], 6.0 * o["a"], 0.5, normflux=o["flux"], subpix=5)
    r = r[np.isfinite(r) & (r > 0.5)]
    return float(np.clip(np.percentile(r, 12), 0.8, 8.0)) if len(r) >= 8 else 1.5


def zeropoint(header, meta):
    """AB magnitude of a source with total flux 1 (in the mosaic's pixel units). None if the header lacks the keywords."""
    if (meta.get("telescope") or "").upper() == "JWST" and header.get("PIXAR_SR"):
        return float(-2.5 * np.log10(1e6 * header["PIXAR_SR"] / 3631.0)), "MJy/sr x PIXAR_SR -> Jy -> AB"
    if header.get("PHOTFLAM") and header.get("PHOTPLAM"):
        return float(-2.5 * np.log10(header["PHOTFLAM"]) - 5 * np.log10(header["PHOTPLAM"]) - 2.408), "PHOTFLAM / PHOTPLAM -> AB"
    return None, "no photometric keywords"


def _inside(px, py, o, R=2.5):
    """Which points (px, py) fall inside any object's ellipse grown R times."""
    if not len(o) or not len(px):
        return np.zeros(len(px), bool)
    from scipy.spatial import cKDTree
    tree, hit = cKDTree(np.c_[px, py]), np.zeros(len(px), bool)
    for x, y, a, b, th in zip(o["x"], o["y"], o["a"], o["b"], o["theta"]):
        idx = tree.query_ball_point([x, y], R * a)
        if not idx:
            continue
        idx = np.asarray(idx)
        dx, dy = px[idx] - x, py[idx] - y
        c, s = np.cos(th), np.sin(th)
        hit[idx[((dx * c + dy * s) / (R * a)) ** 2 + ((-dx * s + dy * c) / (R * max(b, 0.5))) ** 2 < 1]] = True
    return hit


def _fields(o, k=1):
    return {"x": (o["x"] + 0.5) * k - 0.5, "y": (o["y"] + 0.5) * k - 0.5, "a": o["a"] * k, "b": o["b"] * k, "theta": o["theta"].copy(),
            "npix": o["npix"] * k * k, "flag": o["flag"].astype(int)}


def _cat(parts):
    keys = ("x", "y", "a", "b", "theta", "npix", "flag", "mode")
    parts = [p for p in parts if len(p["x"])]
    if not parts:
        return {k: np.array([]) for k in keys}
    return {k: np.concatenate([p[k] for p in parts]) for k in keys}


def run_passes(lin, rms, valid, sky, fwhm, thresholds=None, calibrate=True, fast=False):
    """Cold + hot + low-surface-brightness passes, merged. Returns (catalogue arrays, per-mode report, thresholds used)."""
    mask = ~valid
    thr = dict(thresholds or {})
    rep, parts, covered = {}, [], None
    seg_all = np.zeros(lin.shape, bool)
    for name in ("cold", "hot", "lsb4", "lsb8"):
        m, k = MODES[name], MODES[name]["bin"]
        if k == 1:
            D = _detimg(lin, rms, mask, sky, fwhm * m["kernel"])
            mk = mask
        else:
            # what is left after removing everything already found: only diffuse light can still stand out
            gone = seg_all | mask
            mk = _bin((~gone).astype(np.float32), k) < 0.6
            D = _detimg(_bin(np.where(gone, 0, lin), k), np.maximum(_bin(rms, k) / k, 1e-30), mk, ~mk, max(2.0, fwhm / k * 2) * m["kernel"])
        t = thr.get(name, m["thresh"])
        while calibrate:                                   # cheap, undeblended counts until the noise-only rate is low enough
            n_pos = len(_extract(D, mk, m, t, fast=True)[0])
            neg = _extract(np.ascontiguousarray(-D), mk, m, t, fast=True)[0]       # the flipped image holds noise only
            if len(neg) / max(n_pos, 1) <= MAX_SPURIOUS or n_pos < 20 or t >= m["thresh"] + 3:
                break
            t += 0.5
        o, seg = _extract(D, mk, m, t, fast=fast)
        rejected = calibrate and len(neg) / max(n_pos, 1) > MAX_SPURIOUS
        if rejected:                                       # this pass cannot tell its detections from noise here: use none of them
            o, seg = o[:0], np.zeros_like(seg)
        thr[name] = t
        o = o[(o["a"] > 0) & (o["b"] > 0)]
        fl = _fields(o, k)
        fl["mode"] = np.full(len(o), name)
        keep = np.ones(len(o), bool)
        if covered is not None and len(o):
            keep = ~_inside(fl["x"], fl["y"], covered)
        fl = {kk: v[keep] for kk, v in fl.items()}
        parts.append(fl)
        covered = _cat(parts)
        if k == 1:
            seg_all |= seg > 0
        else:
            import cv2
            up = cv2.resize((seg > 0).astype(np.uint8), (seg.shape[1] * k, seg.shape[0] * k), interpolation=cv2.INTER_NEAREST).astype(bool)
            seg_all[:up.shape[0], :up.shape[1]] |= up
        rep[name] = {"threshold_sigma": round(t, 2), "found": int(len(o)), "kept_new": int(keep.sum()),
                     "spurious_rate": round(len(neg) / max(n_pos, 1), 4) if calibrate else None,
                     "rejected_as_noise": bool(rejected)}
    return _cat(parts), rep, thr


def measure(cat, lin, rms, valid, r_psf, zp, header):
    """Kron photometry, magnitudes, sizes, classes and sky positions for the merged catalogue."""
    n = len(cat["x"])
    if not n:
        return cat
    x, y, a, b, th = (np.ascontiguousarray(cat[k], dtype=np.float64) for k in ("x", "y", "a", "b", "theta"))
    th = np.clip(th, -np.pi / 2 + 1e-6, np.pi / 2 - 1e-6)
    mask = ~valid
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kr, _ = sep.kron_radius(lin, x, y, a, b, th, 6.0, mask=mask)
        kr = np.where(np.isfinite(kr) & (kr > 0), kr, 1.0)
        flux, ferr, _ = sep.sum_ellipse(lin, x, y, a, b, th, 2.5 * kr, subpix=1, err=rms, mask=mask)
        small = kr * np.sqrt(a * b) < 1.75
        if small.any():                                   # too small for a Kron ellipse: fixed circle (the SExtractor rule)
            cf, ce, _ = sep.sum_circle(lin, x[small], y[small], 1.75 * 2.5 / 2, subpix=1, err=rms, mask=mask)
            flux[small], ferr[small] = cf, ce
        r50, _ = sep.flux_radius(lin, x, y, 6.0 * a, 0.5, normflux=np.where(flux > 0, flux, 1.0), subpix=5)
    r50 = np.clip(np.nan_to_num(r50, nan=r_psf), 0.4, None)
    snr = flux / np.maximum(ferr, 1e-30)
    ellip = 1 - b / a
    res = r50 / r_psf
    size_class = np.full(n, "point", dtype="U8")
    lo = STAR_MAX
    for name, hi in SIZE_CLASSES:
        size_class[(res > lo) & (res <= hi)] = name
        lo = max(lo, hi)
    label = np.where(res <= STAR_MAX, "star", np.where(ellip > 0.6, "elongated galaxy", np.where(res > 4.0, "large galaxy", "galaxy")))
    out = dict(cat)
    out.update(flux=flux, flux_err=ferr, snr=snr, r50_px=r50, ellipticity=ellip, size_class=size_class, label=label,
               mag=np.where(flux > 0, zp - 2.5 * np.log10(np.maximum(flux, 1e-30)), np.nan) if zp is not None else np.full(n, np.nan),
               mag_err=np.where(flux > 0, 1.0857 * ferr / np.maximum(flux, 1e-30), np.nan))
    try:
        from astropy.wcs import WCS
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ra, dec = WCS(header).celestial.all_pix2world(x, y, 0)
        out["ra"], out["dec"] = ra, dec
    except Exception:  # noqa: BLE001
        out["ra"] = out["dec"] = np.full(n, np.nan)
    return out


def completeness(lin, rms, valid, smask, fwhm, r_psf, zp, thresholds, existing, grms, per_bin=14, seed=11):
    """Inject fake galaxies (exponential disks blurred by the PSF) of each size class and brightness; count how many
    the same detector, with the same frozen thresholds, finds again. Returns the recovered fraction per bin and the
    magnitude where half are found."""
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(seed)
    H, W = lin.shape
    free = valid & ~smask
    ys, xs = np.nonzero(free[::8, ::8])
    if len(ys) < per_bin * 6 or zp is None:
        return {"skipped": "not enough empty sky" if zp is not None else "no photometric zero point"}
    m5 = zp - 2.5 * np.log10(5 * grms * np.sqrt(np.pi) * 2 * r_psf)          # ~5 sigma for a point source
    steps = np.arange(-4.0, 1.6, 1.0)
    tree0 = cKDTree(np.c_[existing["x"], existing["y"]]) if len(existing["x"]) else None
    out = {"mag_5sigma_point": round(float(m5), 2), "classes": {}}
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    for cname, ratio in (("micro", 1.6), ("small", 3.0), ("medium", 6.0), ("large", 12.0), ("giant", 24.0)):
        r50 = ratio * r_psf
        # a bigger galaxy spreads the same light over more noise, so each size is tested over its own (brighter) range
        mags = np.round(m5 - 2.5 * np.log10(ratio / 1.6) + steps, 2)
        n = per_bin * len(mags)
        pick = rng.choice(len(ys), min(n, len(ys)), replace=False)
        px, py = xs[pick] * 8.0 + rng.uniform(0, 7, len(pick)), ys[pick] * 8.0 + rng.uniform(0, 7, len(pick))
        mag = np.repeat(mags, per_bin)[:len(pick)]
        model = np.zeros_like(lin)
        h = np.hypot(r50, r_psf) / 1.678                     # exponential scale length, PSF folded in
        half = int(min(max(8 * h, 12), 400))
        for x, y, m in zip(px, py, mag):
            x0, x1, y0, y1 = int(max(0, x - half)), int(min(W, x + half)), int(max(0, y - half)), int(min(H, y + half))
            q, pa = rng.uniform(0.4, 1.0), rng.uniform(0, np.pi)
            dx, dy = xx[y0:y1, x0:x1] - x, yy[y0:y1, x0:x1] - y
            u, v = dx * np.cos(pa) + dy * np.sin(pa), (-dx * np.sin(pa) + dy * np.cos(pa)) / q
            prof = np.exp(-np.hypot(u, v) / h)
            model[y0:y1, x0:x1] += prof * (10 ** (-0.4 * (m - zp)) / max(float(prof.sum()), 1e-30))
        cat, _, _ = run_passes(np.ascontiguousarray(np.where(valid, lin + model, 0), dtype=np.float32), rms, valid, free, fwhm, thresholds, calibrate=False)       # deblended exactly like the real catalogue
        found = np.zeros(len(px), bool)
        radius, chance = float(np.clip(0.5 * r50, 2.5, 15.0)), 0.0
        if len(cat["x"]):
            cx, cy = cat["x"], cat["y"]
            if tree0 is not None:                               # ignore what was there before the injection
                d0, _ = tree0.query(np.c_[cx, cy])
                new = d0 > 1.5
                cx, cy = cx[new], cy[new]
            if len(cx):
                t = cKDTree(np.c_[cx, cy])
                found = t.query(np.c_[px, py])[0] < radius
                # how often an empty spot "matches" by accident in this field (crowding): removed from the fractions
                rnd = rng.choice(len(ys), min(2000, len(ys)), replace=False)
                chance = float((t.query(np.c_[xs[rnd] * 8.0 + 3.5, ys[rnd] * 8.0 + 3.5])[0] < radius).mean())
        fix = lambda f: max(0.0, (f - chance) / max(1 - chance, 1e-9))  # noqa: E731
        frac = [round(fix(float(found[mag == m].mean())), 3) if (mag == m).any() else None for m in mags]
        lim = None
        for i in range(len(mags) - 1):                          # brightest crossing of 50%
            a, b = frac[i], frac[i + 1]
            if a is not None and b is not None and a >= 0.5 > b:
                lim = float(mags[i] + (a - 0.5) / max(a - b, 1e-9) * (mags[i + 1] - mags[i]))
                break
        if lim is None and all(f is not None and f >= 0.5 for f in frac):
            lim = float(mags[-1])                               # still above half at the faintest bin tested
        out["classes"][cname] = {"r50_px": round(float(r50), 2), "mags": [float(m) for m in mags], "recovered": frac, "chance_match_rate": round(chance, 4), "mag_50": round(lim, 2) if lim is not None else None,
                                 "limit_is_lower_bound": bool(lim is not None and frac[-1] >= 0.5)}
    return out


def detect(path, with_completeness=True):
    """FITS mosaic -> (catalogue dict of arrays, report)."""
    T, t = {}, time.time()
    d = fits_loader.load(path)
    sci, valid, meta, header = d["sci"], d["valid"], d["meta"], d["header"]
    bg = background.estimate(sci, valid)
    lin = np.ascontiguousarray(np.where(valid, sci - bg["back"], 0), dtype=np.float32)
    rms = np.ascontiguousarray(np.maximum(bg["rms"], 1e-30), dtype=np.float32)
    T["background"] = round(time.time() - t, 1)
    t = time.time()
    r_psf = psf_r50(lin, rms, ~valid)
    fwhm = 2 * r_psf
    zp, zp_how = zeropoint(header, meta)
    sky = valid & ~bg["source_mask"]
    cat, modes, thr = run_passes(lin, rms, valid, sky, fwhm)
    cat = measure(cat, lin, rms, valid, r_psf, zp, header)
    n = len(cat["x"])
    if n:                                                  # a real source has positive flux and a believable S/N
        ok = (cat["flux"] > 0) & (cat["snr"] >= 3.0)
        cat = {k: v[ok] for k, v in cat.items()}
    T["detect"] = round(time.time() - t, 1)
    n = len(cat["x"])
    pix = meta.get("pixscale_arcsec")
    rep = {"version": VERSION, "meta": meta, "n_sources": int(n), "psf_r50_px": round(r_psf, 2),
           "psf_fwhm_arcsec": round(fwhm * pix, 3) if pix else None, "zeropoint_ab": round(zp, 3) if zp is not None else None, "zeropoint_from": zp_how,
           "modes": modes, "max_spurious_rate_allowed": MAX_SPURIOUS,
           "counts": {k: int(v) for k, v in zip(*np.unique(cat["label"], return_counts=True))} if n else {},
           "size_classes": {k: int(v) for k, v in zip(*np.unique(cat["size_class"], return_counts=True))} if n else {},
           "by_mode": {k: int(v) for k, v in zip(*np.unique(cat["mode"], return_counts=True))} if n else {},
           "mag_range": [round(float(np.nanmin(cat["mag"])), 2), round(float(np.nanmax(cat["mag"])), 2)] if n and zp is not None else None,
           "shape": list(lin.shape)}
    if with_completeness and n:
        t = time.time()
        rep["completeness"] = completeness(lin, rms, valid, bg["source_mask"], fwhm, r_psf, zp, thr, cat, bg["global_rms"])
        rep["completeness_note"] = ("Injected into the background-subtracted image; the background model is not re-fitted, "
                                    "so for the largest sizes this is an upper limit (see the restoration report).")
        T["completeness"] = round(time.time() - t, 1)
    rep["timings_s"] = T
    return cat, rep


def write_table(cat, rep, path):
    """The catalogue as an ECSV table (readable by astropy, TOPCAT, pandas) with the report in its header."""
    import json
    from astropy.table import Table
    cols = ("ra", "dec", "x", "y", "mag", "mag_err", "flux", "flux_err", "snr", "r50_px", "a", "b", "theta", "ellipticity", "size_class", "label", "mode", "flag")
    t = Table({c: cat[c] for c in cols if c in cat})
    for c in t.colnames:
        if t[c].dtype.kind == "f":
            t[c].info.format = ".6f" if c in ("ra", "dec") else ".4g"
    t.meta["report"] = json.loads(json.dumps(rep, default=str))
    t.meta["note"] = "x, y are 0-based pixel positions in the science mosaic; mags are AB, Kron apertures."
    t.write(path, format="ascii.ecsv", overwrite=True)
    return path


if __name__ == "__main__":
    import json
    import sys
    from . import catalog_db
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(catalog_db.run(int(sys.argv[1])), indent=1, default=str))
