"""Did processing keep the science? Source extraction before vs after, on the LINEAR data (never on display pixels).

Reference: the original pixels minus one global sky level (what the old renderer effectively showed).
Processed:  the original pixels minus our background model.
Background subtraction must not change compact sources (same count, same flux, same centroid) and must not eat
large galaxies (their flux inside an aperture should stay close to the reference). Flatter sky is the goal.
"""
import numpy as np
import sep

CHECKS = {"recovered_min": 0.95, "centroid_shift_max_px": 0.15, "flux_ratio_tol": 0.05, "large_flux_ratio_min": 0.85}


def _extract(img, rms, valid, thresh=5.0):
    return sep.extract(np.ascontiguousarray(img, dtype=np.float32), thresh, err=rms, mask=~valid, minarea=5, deblend_cont=0.005)


def _match(a, b, r=1.5):
    if not len(a) or not len(b):
        return np.array([], int), np.array([], int)
    from scipy.spatial import cKDTree
    d, j = cKDTree(np.c_[b["x"], b["y"]]).query(np.c_[a["x"], a["y"]], distance_upper_bound=r)
    ok = np.isfinite(d)
    return np.nonzero(ok)[0], j[ok]


def compare(sci, lin, valid, rms_map, global_rms, sky_fraction=None):
    ref = (sci - np.median(sci[valid])).astype(np.float32)
    ref[~valid] = 0
    o_ref = _extract(ref, global_rms, valid)
    o_new = _extract(lin, global_rms, valid)
    ia, ib = _match(o_ref, o_new)
    out = {"n_reference": int(len(o_ref)), "n_processed": int(len(o_new)), "matched": int(len(ia)),
           "recovered": round(len(ia) / max(len(o_ref), 1), 4), "new": int(len(o_new) - len(ib))}
    if len(ia):
        dx = o_ref["x"][ia] - o_new["x"][ib]
        dy = o_ref["y"][ia] - o_new["y"][ib]
        out["centroid_shift_median_px"] = round(float(np.median(np.hypot(dx, dy))), 4)
        # compact-source flux: small circular aperture, same position in both
        x, y = o_ref["x"][ia], o_ref["y"][ia]
        f0, _, _ = sep.sum_circle(ref, x, y, 3.0, mask=~valid)
        f1, _, _ = sep.sum_circle(np.ascontiguousarray(lin), x, y, 3.0, mask=~valid)
        good = f0 > 10 * global_rms * 3
        out["flux_ratio_median"] = round(float(np.median(f1[good] / f0[good])), 4) if good.any() else None
        # large sources: the 10 biggest, elliptical aperture 3x their size
        big = np.argsort(-o_ref["npix"][ia])[:10]
        k = ia[big]
        a, b, th = o_ref["a"][k] * 3, o_ref["b"][k] * 3, o_ref["theta"][k]
        F0, _, _ = sep.sum_ellipse(ref, o_ref["x"][k], o_ref["y"][k], a, b, th, r=1.0, mask=~valid)
        F1, _, _ = sep.sum_ellipse(np.ascontiguousarray(lin), o_ref["x"][k], o_ref["y"][k], a, b, th, r=1.0, mask=~valid)
        ok = F0 > 0
        out["large_flux_ratio_median"] = round(float(np.median(F1[ok] / F0[ok])), 4) if ok.any() else None
        out["large_sources"] = [{"x": round(float(o_ref["x"][i]), 1), "y": round(float(o_ref["y"][i]), 1), "npix": int(o_ref["npix"][i]),
                                 "flux_ratio": round(float(f1_ / f0_), 4) if f0_ > 0 else None}
                                for i, f0_, f1_ in zip(k, F0, F1)]
    c = CHECKS
    out["checks"] = {
        "sources_kept": out["recovered"] >= c["recovered_min"],
        "centroids_stable": out.get("centroid_shift_median_px", 0) <= c["centroid_shift_max_px"],
        "compact_flux_kept": out.get("flux_ratio_median") is None or abs(out["flux_ratio_median"] - 1) <= c["flux_ratio_tol"],
        "large_galaxies_kept": out.get("large_flux_ratio_median") is None or out["large_flux_ratio_median"] >= c["large_flux_ratio_min"],
    }
    out["informational"] = out.pop("checks")      # the reference (global median) is biased in crowded fields: not ground truth
    out["checks"] = {"enough_sky_for_background": sky_fraction is None or sky_fraction >= 0.2}
    out["passed"] = all(out["checks"].values())
    return out


def injection_test(sci, valid, lin, smask, rms, estimate_bg, n_point=60, n_disk=10, seed=3, fwhm=2.5):
    """Inject sources of KNOWN flux into empty sky, rerun the background step, measure what survives.

    Points: Gaussians (FWHM ~ the PSF). Disks: large faint exponential galaxies (scale 10-40 px, central surface
    brightness ~1-2 sigma per pixel), the kind a background model is most likely to swallow. The difference
    (processed with injections) - (processed without) isolates the injected light, so the field's own content cancels.
    """
    rng = np.random.default_rng(seed)
    H, W = sci.shape
    free = valid & ~smask
    ys, xs = np.nonzero(free[::4, ::4])
    if len(ys) < n_point + n_disk:
        return {"skipped": "not enough empty sky"}
    pick = rng.choice(len(ys), n_point + n_disk, replace=False)
    pos = np.c_[xs[pick] * 4, ys[pick] * 4].astype(float)
    model = np.zeros_like(sci)
    yy, xx = np.mgrid[0:H, 0:W]
    sig = fwhm / 2.355
    pts, disks = [], []
    for i, (x, y) in enumerate(pos):
        x0, x1, y0, y1 = int(max(0, x - 160)), int(min(W, x + 160)), int(max(0, y - 160)), int(min(H, y + 160))
        sx, sy = xx[y0:y1, x0:x1] - x, yy[y0:y1, x0:x1] - y
        r2 = sx * sx + sy * sy
        if i < n_point:
            f = rng.uniform(15, 60) * rms * 2 * np.sqrt(np.pi) * sig        # 15-60 sigma peak-ish
            model[y0:y1, x0:x1] += f * np.exp(-r2 / (2 * sig * sig)) / (2 * np.pi * sig * sig)
            pts.append((x, y, f))
        else:
            h = rng.uniform(10, 40)
            mu0 = rng.uniform(1.0, 2.0) * rms
            model[y0:y1, x0:x1] += mu0 * np.exp(-np.sqrt(r2) / h)
            disks.append((x, y, h, mu0))
    inj = np.where(valid, sci + model, 0).astype(np.float32)
    bg2 = estimate_bg(inj, valid)
    lin2 = (inj - bg2["back"]).astype(np.float32)
    lin2[~valid] = 0
    diff = np.ascontiguousarray(lin2 - lin)
    mod = np.ascontiguousarray(model)
    px, py = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
    rec_p, _, _ = sep.sum_circle(diff, px, py, 2 * fwhm)
    true_p, _, _ = sep.sum_circle(mod, px, py, 2 * fwhm)
    dx, dy = np.array([d[0] for d in disks]), np.array([d[1] for d in disks])
    rr = np.array([3 * d[2] for d in disks])
    rec_d, _, _ = sep.sum_circle(diff, dx, dy, rr)
    true_d, _, _ = sep.sum_circle(mod, dx, dy, rr)
    pr, dr = rec_p / true_p, rec_d / true_d
    out = {"points": int(len(pts)), "disks": int(len(disks)),
           "point_flux_ratio_median": round(float(np.median(pr)), 4), "point_flux_ratio_p10": round(float(np.percentile(pr, 10)), 4),
           "disk_flux_ratio_median": round(float(np.median(dr)), 4), "disk_flux_ratio_min": round(float(dr.min()), 4),
           "disk_scale_px": [round(float(d[2]), 1) for d in disks], "disk_flux_ratios": [round(float(v), 3) for v in dr]}
    out["checks"] = {"point_flux_kept": abs(out["point_flux_ratio_median"] - 1) <= 0.03,
                     "faint_large_galaxies_kept": out["disk_flux_ratio_median"] >= 0.85}
    out["passed"] = all(out["checks"].values())
    return out
