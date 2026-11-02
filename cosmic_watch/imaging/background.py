"""Background model that leaves galaxies alone (SEP, two passes, multi-scale source mask).

A plain mesh background (median per box) treats a big galaxy halo as "sky" and subtracts it. So:
  1. rough background -> detect sources at full resolution AND on a 4x-binned image (finds large, faint halos that
     are invisible pixel-by-pixel), dilate the union into a source mask;
  2. re-estimate the background with every source pixel masked;
  3. the mesh size adapts to the sources: several times the size of the largest typical galaxy, so no single box is
     dominated by one object.
Returns the smooth background, the per-pixel noise (rms) map and diagnostics. Nothing is written into the pixels
here; subtraction happens in the pipeline.
"""
import cv2
import numpy as np
import sep

sep.set_extract_pixstack(20_000_000)
sep.set_sub_object_limit(4096)


def _bin(a, k):
    h, w = (a.shape[0] // k) * k, (a.shape[1] // k) * k
    return a[:h, :w].reshape(h // k, k, w // k, k).mean(axis=(1, 3))


def _seg_mask(img, rms, thresh, minarea, mask):
    try:
        _, seg = sep.extract(img, thresh, err=rms, mask=mask, minarea=minarea, segmentation_map=True, deblend_cont=1.0)
        return seg > 0
    except Exception:  # noqa: BLE001  (pixel stack overflow on huge crowded fields: fall back to a threshold mask)
        return (img > thresh * rms) & ~mask


MIN_GOOD = 0.10           # a box needs >=10% unmasked sky pixels (SEP's built-in rule is 50%: unstable in crowded fields)


def mesh_background(data, use, mesh, clip=3.0, filt=3):
    """Sigma-clipped median + MAD per box over pixels where `use`; boxes with too little sky are filled from their
    neighbours; median-filtered on the box grid; bicubic upsample. Returns (back, rms) full-size float32."""
    H, W = data.shape
    ny, nx = -(-H // mesh), -(-W // mesh)
    pad = np.full((ny * mesh, nx * mesh), np.nan, np.float32)
    pad[:H, :W] = np.where(use, data, np.nan)
    boxes = pad.reshape(ny, mesh, nx, mesh).transpose(0, 2, 1, 3).reshape(ny, nx, -1)
    frac = np.isfinite(boxes).mean(axis=2)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(boxes, axis=2)
        mad = np.nanmedian(np.abs(boxes - med[..., None]), axis=2) * 1.4826
        clipped = np.where(np.abs(boxes - med[..., None]) <= clip * mad[..., None], boxes, np.nan)
        med = np.nanmedian(clipped, axis=2)
        mad = np.nanmedian(np.abs(clipped - med[..., None]), axis=2) * 1.4826
    bad = (frac < MIN_GOOD) | ~np.isfinite(med)
    if bad.all():
        raise ValueError("no box has enough sky for a background estimate")
    if bad.any():                                       # fill from the nearest good boxes
        from scipy.ndimage import distance_transform_edt
        _, (iy, ix) = distance_transform_edt(bad, return_indices=True)
        med, mad = med[iy, ix], mad[iy, ix]
    if filt > 1 and min(ny, nx) >= filt:
        from scipy.ndimage import median_filter
        med, mad = median_filter(med, filt, mode="nearest"), median_filter(mad, filt, mode="nearest")
    # box centres -> pixels: upsample the grid with edge padding so centres land at (i + 0.5) * mesh
    def up(g):
        gp = np.pad(g.astype(np.float32), 1, mode="edge")
        full = cv2.resize(gp, ((nx + 2) * mesh, (ny + 2) * mesh), interpolation=cv2.INTER_CUBIC)
        return np.ascontiguousarray(full[mesh: mesh + H, mesh: mesh + W])
    return up(med), up(mad), float(np.mean(bad))


MAX_MASKED = 0.6          # if sources cover more than this, thresholds are raised (never estimate sky from 2% of pixels)


def estimate(sci, valid, mesh=None, dilate=None, coarse=True, thresh=3.0, coarse_thresh=2.5, adapt=True):
    """sci: float32 (C-contiguous), valid: bool. mesh: box size in px (None = adaptive). Returns dict."""
    data = np.ascontiguousarray(sci, dtype=np.float32)
    bad = ~valid
    m0 = mesh or max(64, min(256, min(data.shape) // 24))
    b0 = sep.Background(data, mask=bad, bw=m0, bh=m0, fw=3, fh=3)
    rms0 = b0.rms()
    resid = data - b0.back()
    for _ in range(4):
        smask = _seg_mask(resid, rms0, thresh, 5, bad)
        if not adapt or smask[valid].mean() <= MAX_MASKED:
            break
        thresh *= 1.5
    if coarse:                                          # large / low-surface-brightness structure
        k = 4
        rb = np.ascontiguousarray(_bin(np.where(valid, resid, 0), k), dtype=np.float32)
        vb = _bin(valid.astype(np.float32), k) > 0.5
        sb = sep.Background(rb, mask=~vb, bw=max(16, m0 // k), bh=max(16, m0 // k), fw=3, fh=3)
        rr = np.ascontiguousarray(rb - sb.back())
        for _ in range(4):
            cm = _seg_mask(rr, sb.rms(), coarse_thresh, 16, ~vb)
            if not adapt or cm[vb].mean() <= MAX_MASKED / 2:
                break
            coarse_thresh *= 1.5
        cm = cv2.resize(cm.astype(np.uint8), (cm.shape[1] * k, cm.shape[0] * k), interpolation=cv2.INTER_NEAREST).astype(bool)
        full = np.zeros_like(smask)
        full[:cm.shape[0], :cm.shape[1]] = cm
        smask |= full
    # adaptive mesh from the size of what was found
    if mesh is None:
        n, lab, stats, _ = cv2.connectedComponentsWithStats(smask.astype(np.uint8), connectivity=8)
        sizes = np.sqrt(stats[1:, cv2.CC_STAT_AREA]) if n > 1 else np.array([8.0])
        mesh = int(np.clip(2 * np.percentile(sizes, 99), 64, 256))   # validated: 4x (295 px) was unstable under injection
    d = dilate if dilate is not None else 3      # small fixed halo; large objects already come big from the coarse pass
    if d:
        smask = cv2.dilate(smask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1))).astype(bool)
    sky = valid & ~smask
    back, rms, filled = mesh_background(data, sky, mesh)
    gb = float(np.median(data[sky])) if sky.any() else 0.0
    grms = float(np.median(rms[sky])) if sky.any() else 1.0              # local noise, not inflated by gradients
    return {"back": back, "rms": rms, "source_mask": smask, "mesh": mesh, "dilate": d, "filled_boxes": round(filled, 3),
            "global_back": gb, "global_rms": grms,
            "thresholds": [round(thresh, 2), round(coarse_thresh, 2) if coarse else None],
            "masked_fraction": round(float(smask[valid].mean()) if valid.any() else 0.0, 4),
            "sky_pixels": int(sky.sum())}


def uniformity(lin, valid, smask, box=128):
    """Spread of the sky level across the image, in units of the noise (lower = flatter background)."""
    sky = valid & ~smask
    meds = []
    for y in range(0, lin.shape[0] - box + 1, box):
        for x in range(0, lin.shape[1] - box + 1, box):
            s = sky[y:y + box, x:x + box]
            if s.mean() > 0.5:
                meds.append(np.median(lin[y:y + box, x:x + box][s]))
    if len(meds) < 4:
        return None
    meds = np.array(meds)
    noise = 1.4826 * np.median(np.abs(lin[sky] - np.median(lin[sky]))) if sky.any() else 1.0
    return round(float(np.std(meds) / max(noise, 1e-12)), 4)
