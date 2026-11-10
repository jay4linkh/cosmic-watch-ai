"""Sunspots, sunspot groups, active regions and coronal holes on SDO full-disk images, in solar coordinates.

What is measured, not just outlined:
  * the limb is fitted (visible light) or taken from the ephemeris (EUV, where the corona hides the edge);
  * limb darkening is fitted and divided out, so a spot near the edge is judged like one at the centre;
  * every feature gets a heliographic latitude / longitude (B0 and L0 from the date) and an area corrected for
    foreshortening, in millionths of the solar hemisphere (MSH), the unit NOAA uses;
  * spots are split into umbra and penumbra and clustered into groups, giving a sunspot number (10 x groups + spots);
  * the groups are checked against NOAA's numbered active regions for the same day.
These are 1024-px, 8-bit browse images: fine for positions, groups and big areas, not for sub-arcsecond work.
"""
import math
from datetime import datetime, timedelta, timezone

import cv2
import numpy as np

D2R = math.pi / 180
AIA_ARCSEC_PX = 2.4                 # SDO 1024-px browse images
MIN_MU = 0.25                       # beyond ~75 deg from disk centre the foreshortening correction blows up
GROUP_DEG = 7.0                     # spots closer than this (great circle) belong to one group


def _when(when):
    if isinstance(when, datetime):
        return when if when.tzinfo else when.replace(tzinfo=timezone.utc)
    if when:
        try:
            return datetime.fromisoformat(str(when).replace("Z", "+00:00").replace(" ", "T")).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return datetime.now(timezone.utc)


def ephemeris(when=None):
    """B0 (tilt of the Sun's north pole toward us), L0 (Carrington longitude of disk centre), apparent radius. Meeus ch. 25, 29."""
    t = _when(when)
    jd = t.timestamp() / 86400.0 + 2440587.5
    T = (jd - 2451545.0) / 36525
    M = (357.52911 + 35999.05029 * T) * D2R
    C = (1.914602 - 0.004817 * T) * math.sin(M) + 0.019993 * math.sin(2 * M) + 0.000289 * math.sin(3 * M)
    lam = (280.46646 + 36000.76983 * T + C - 0.00569) * D2R
    e = 0.016708634 - 0.000042037 * T
    dist = 1.000001018 * (1 - e * e) / (1 + e * math.cos(M + C * D2R))
    inc, K = 7.25 * D2R, (73.6667 + 1.3958333 * (jd - 2396758) / 36525) * D2R
    theta = ((jd - 2398220) * 360 / 25.38) % 360
    b0 = math.asin(math.sin(lam - K) * math.sin(inc))
    eta = math.atan2(-math.sin(lam - K) * math.cos(inc), -math.cos(lam - K))
    return {"b0_deg": b0 / D2R, "l0_deg": (eta / D2R - theta) % 360, "radius_arcsec": 959.63 / dist, "dist_au": dist, "time": t}


def _limb_fit(gray):
    """Centre and radius of the visible-light disk: a circle fitted to the steepest edge along 360 rays."""
    h, w = gray.shape
    cx, cy, r0 = (w - 1) / 2, (h - 1) / 2, 0.47 * w
    g = cv2.GaussianBlur(gray, (0, 0), 1.5)
    for _ in range(2):
        pts = []
        for a in np.linspace(0, 2 * np.pi, 360, endpoint=False):
            rr = np.arange(max(5, r0 - 40), min(r0 + 40, 0.7 * w), 0.5)
            xs, ys = cx + rr * np.cos(a), cy + rr * np.sin(a)
            ok = (xs >= 0) & (xs < w - 1) & (ys >= 0) & (ys < h - 1)
            if ok.sum() < 20:
                continue
            v = g[ys[ok].astype(int), xs[ok].astype(int)]
            k = int(np.argmin(np.gradient(v)))
            pts.append((xs[ok][k], ys[ok][k]))
        p = np.array(pts)
        if len(p) < 60:
            break
        A = np.c_[2 * p[:, 0], 2 * p[:, 1], np.ones(len(p))]
        sol = np.linalg.lstsq(A, (p ** 2).sum(1), rcond=None)[0]
        res = np.abs(np.hypot(p[:, 0] - sol[0], p[:, 1] - sol[1]) - math.sqrt(sol[2] + sol[0] ** 2 + sol[1] ** 2))
        keep = res < max(1.5, 3 * np.median(res))               # drop rays that hit a spot at the limb
        sol = np.linalg.lstsq(A[keep], (p[keep] ** 2).sum(1), rcond=None)[0]
        cx, cy, r0 = float(sol[0]), float(sol[1]), float(math.sqrt(sol[2] + sol[0] ** 2 + sol[1] ** 2))
    return (cx, cy), r0


def _grid(shape, centre, r):
    """Disk coordinates in solar radii: x toward solar west (right), y toward north (up); rho and mu = cos(angle from centre)."""
    yy, xx = np.mgrid[:shape[0], :shape[1]].astype(np.float32)
    x, y = (xx - centre[0]) / r, (centre[1] - yy) / r
    rho = np.hypot(x, y)
    return x, y, rho, np.sqrt(np.clip(1 - rho * rho, 0, 1))


def heliographic(x, y, b0_deg):
    """Disk position (solar radii) -> latitude and longitude from the central meridian (west positive), degrees."""
    rho = np.clip(np.hypot(x, y), 1e-9, 1)
    cos_c, b0 = np.sqrt(1 - rho * rho), b0_deg * D2R
    lat = np.arcsin(np.clip(cos_c * math.sin(b0) + y * math.cos(b0), -1, 1))
    lon = np.arctan2(x, cos_c * math.cos(b0) - y * math.sin(b0))
    return lat / D2R, lon / D2R


def _msh(area_px, mu, r):
    """Projected pixels -> millionths of the solar hemisphere, corrected for foreshortening."""
    return float(area_px / max(mu, MIN_MU) / (2 * math.pi * r * r) * 1e6)


def _loc(lat, lon):
    return f"{'N' if lat >= 0 else 'S'}{abs(round(lat)):02d}{'W' if lon >= 0 else 'E'}{abs(round(lon)):02d}"


def _sep_deg(a, b):
    la, lb, d = a["lat"] * D2R, b["lat"] * D2R, (a["lon"] - b["lon"]) * D2R
    return math.acos(max(-1, min(1, math.sin(la) * math.sin(lb) + math.cos(la) * math.cos(lb) * math.cos(d)))) / D2R


def _components(mask, min_area):
    n, lab, st, cen = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return [(i, lab, st[i], cen[i]) for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= min_area]


def _group(spots):
    """Single-linkage clustering of spots on the sphere."""
    parent = list(range(len(spots)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(len(spots)):
        for j in range(i + 1, len(spots)):
            if _sep_deg(spots[i], spots[j]) < GROUP_DEG:
                parent[find(i)] = find(j)
    groups = {}
    for i, s in enumerate(spots):
        groups.setdefault(find(i), []).append(s)
    out = []
    for members in groups.values():
        wts = np.array([m["area_msh"] for m in members]) + 1e-6
        g = {"lat": float(np.average([m["lat"] for m in members], weights=wts)), "lon": float(np.average([m["lon"] for m in members], weights=wts)),
             "x": float(np.average([m["x"] for m in members], weights=wts)), "y": float(np.average([m["y"] for m in members], weights=wts)),
             "n_spots": len(members), "area_msh": round(sum(m["area_msh"] for m in members), 1),
             "umbra_fraction": round(sum(m["umbra_px"] for m in members) / max(sum(m["area"] for m in members), 1), 2),
             "extent_deg": round(max((_sep_deg(a, b) for a in members for b in members), default=0.0), 1)}
        g["location"] = _loc(g["lat"], g["lon"])
        out.append(g)
    return sorted(out, key=lambda g: -g["area_msh"])


def match_noaa(groups, noaa, when, max_deg=10.0):
    """Pair our groups with NOAA's numbered regions (rotated from their listed positions to the image time)."""
    if not noaa or not noaa.get("regions"):
        return None
    t = _when(when)
    try:
        # NOAA lists each day's positions as of 24:00 UT of that date (checked against these images: with that
        # epoch four regions agree to ~2 deg; with 00:00 UT they are all off by one day of rotation)
        t0 = datetime.fromisoformat(noaa["date"]).replace(tzinfo=timezone.utc) + timedelta(days=1)
    except (TypeError, ValueError):
        return None
    days = (t - t0).total_seconds() / 86400
    if abs(days) > 2.5:
        return None                                        # a stale region list says nothing about this image
    regs = []
    for r in noaa["regions"]:
        loc = r.get("location") or ""
        if len(loc) < 6 or not r.get("spots"):
            continue
        try:
            lat = int(loc[1:3]) * (1 if loc[0] == "N" else -1)
            lon = int(loc[4:6]) * (1 if loc[3] == "W" else -1) + 13.2 * days          # the Sun turns ~13.2 deg/day as seen from Earth
        except ValueError:
            continue
        if abs(lon) <= 80:
            regs.append({"region": r["region"], "lat": lat, "lon": lon, "spots": r["spots"], "area": r.get("area")})
    pairs, offs = [], []
    for g in groups:
        g.pop("noaa", None)
    for r in sorted(regs, key=lambda r: -(r["area"] or 0)):
        cand = [(_sep_deg(g, r), g) for g in groups if "noaa" not in g]
        if cand:
            d, g = min(cand, key=lambda c: c[0])
            if d <= max_deg:
                g["noaa"] = r["region"]
                g["noaa_offset_deg"] = round(d, 1)
                pairs.append(r["region"])
                offs.append(d)
    return {"noaa_date": noaa["date"], "noaa_visible_with_spots": len(regs), "noaa_recovered": len(pairs),
            "our_groups": len(groups), "ours_without_noaa": sum(1 for g in groups if "noaa" not in g),
            "missed": [r["region"] for r in regs if r["region"] not in pairs],
            "median_offset_deg": round(float(np.median(offs)), 1) if offs else None}


def sunspots(path, when=None, noaa=None):
    """Dark spots on the visible-light (HMI continuum) disk -> spots, groups, sunspot number, areas in MSH."""
    bgr = cv2.imread(str(path))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    eph = ephemeris(when)
    centre, r = _limb_fit(gray)
    x, y, rho, mu = _grid(gray.shape, centre, r)
    disk = rho < 0.985
    # limb darkening: I = I0 (1 - u (1 - mu) - v (1 - mu)^2), fitted to the quiet photosphere (spots clipped out)
    fit = disk & (rho < 0.97)
    A = np.c_[np.ones(fit.sum()), 1 - mu[fit], (1 - mu[fit]) ** 2]
    v = gray[fit]
    keep = np.ones(len(v), bool)
    for _ in range(3):
        coef = np.linalg.lstsq(A[keep], v[keep], rcond=None)[0]
        res = v - A @ coef
        keep = np.abs(res) < 2.5 * 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep])) + 1e-6)
    model = np.maximum(coef[0] + coef[1] * (1 - mu) + coef[2] * (1 - mu) ** 2, 1.0)
    contrast = np.where(disk, gray / model, 1.0)
    noise = float(1.4826 * np.median(np.abs(contrast[fit] - 1)))
    pen_t, umb_t = min(0.92, 1 - 6 * noise), 0.62
    inner = rho < 0.97                                                   # the last 3% is too foreshortened and noisy
    pen = cv2.morphologyEx(((contrast < pen_t) & inner).astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)).astype(bool)
    spots = []
    for i, lab, st, cen in _components(pen, 4):
        m = lab == i
        sx, sy = float(x[m].mean()), float(y[m].mean())
        lat, lon = heliographic(np.array(sx), np.array(sy), eph["b0_deg"])
        area = int(st[cv2.CC_STAT_AREA])
        spots.append({"label": "sunspot", "x": float(cen[0]), "y": float(cen[1]), "a": st[cv2.CC_STAT_WIDTH] / 2, "b": st[cv2.CC_STAT_HEIGHT] / 2,
                      "theta": 0.0, "score": float(1 - contrast[m].mean()), "area": area, "umbra_px": int((m & (contrast < umb_t)).sum()),
                      "lat": float(lat), "lon": float(lon), "area_msh": round(_msh(area, float(mu[m].mean()), r), 1)})
    spots.sort(key=lambda d: -d["area"])
    groups = _group(spots)
    stats = {"n_sunspots": len(spots), "spot_area_px": int(sum(d["area"] for d in spots)),
             "n_groups": len(groups), "sunspot_number": 10 * len(groups) + len(spots), "spot_area_msh": round(sum(d["area_msh"] for d in spots), 1),
             "groups": [{k: (round(v, 2) if isinstance(v, float) else v) for k, v in g.items()} for g in groups[:20]],
             "b0_deg": round(eph["b0_deg"], 2), "l0_deg": round(eph["l0_deg"], 2), "contrast_noise": round(noise, 4),
             "limb_darkening": [round(float(-coef[1] / coef[0]), 3), round(float(-coef[2] / coef[0]), 3)],
             "radius_arcsec_ephemeris": round(eph["radius_arcsec"], 1), "arcsec_per_px": round(eph["radius_arcsec"] / r, 3),
             "disk_radius_px": round(r, 1), "disk_center_px": [round(centre[0], 1), round(centre[1], 1)], "size": [gray.shape[1], gray.shape[0]],
             "detector": "sun-2"}
    m = match_noaa(groups, noaa, when)
    if m:
        stats["noaa"] = m
        stats["groups"] = [{k: (round(v, 2) if isinstance(v, float) else v) for k, v in g.items()} for g in groups[:20]]
    return bgr, spots, stats


def active_regions(path, n_sigma: float = 3.0, when=None, holes=False):
    """Bright active regions in the extreme-UV corona (AIA), and (193 A, holes=True) dark coronal holes.

    The chromosphere (304 A) is mottled all over, so call it with a higher n_sigma.
    """
    bgr = cv2.imread(str(path))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    eph = ephemeris(when)
    h, w = gray.shape
    centre, r = ((w - 1) / 2, (h - 1) / 2), eph["radius_arcsec"] / AIA_ARCSEC_PX * w / 1024     # the corona hides the limb: use the ephemeris
    x, y, rho, mu = _grid(gray.shape, centre, r)
    disk = rho < 0.98
    vals = gray[disk].astype(np.float32)
    med = float(np.median(vals))
    sigma = 1.4826 * float(np.median(np.abs(vals - med))) + 1e-3
    thr = med + n_sigma * sigma
    bright = cv2.morphologyEx(((gray > thr) & disk).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    bright = cv2.dilate(bright, np.ones((9, 9), np.uint8)).astype(bool) & disk       # merge loop systems into one region

    def describe(i, lab, st, cen, label, score):
        m = lab == i
        lat, lon = heliographic(np.array(float(x[m].mean())), np.array(float(y[m].mean())), eph["b0_deg"])
        area = int(st[cv2.CC_STAT_AREA])
        return {"label": label, "x": float(cen[0]), "y": float(cen[1]), "a": st[cv2.CC_STAT_WIDTH] / 2, "b": st[cv2.CC_STAT_HEIGHT] / 2, "theta": 0.0,
                "score": score(m), "area": area, "lat": round(float(lat), 1), "lon": round(float(lon), 1), "location": _loc(float(lat), float(lon)),
                "area_msh": round(_msh(area, float(mu[m].mean()), r), 0)}
    regs = sorted((describe(*c, "active region", lambda m: float(gray[m].mean())) for c in _components(bright, 150)), key=lambda d: -d["area"])
    stats = {"n_active_regions": len(regs), "activity_index": round(100.0 * float(((gray > thr) & disk).sum()) / float(disk.sum()), 2),
             "peak_brightness": int(gray[disk].max()), "b0_deg": round(eph["b0_deg"], 2), "l0_deg": round(eph["l0_deg"], 2),
             "regions": [{k: d[k] for k in ("location", "lat", "lon", "area_msh")} for d in regs[:12]],
             "disk_radius_px": round(r, 1), "disk_center_px": [round(centre[0], 1), round(centre[1], 1)], "size": [w, h], "detector": "sun-2"}
    if holes:
        # coronal holes: large areas much darker than the quiet corona (open magnetic field, source of fast solar wind).
        # Filament channels are also dark, so only big, compact regions are kept; this is a candidate list.
        inner = rho < 0.93
        sm = cv2.GaussianBlur(gray, (0, 0), 3)
        dark = cv2.morphologyEx(((sm < 0.4 * med) & inner).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        hs = sorted((describe(*c, "coronal hole", lambda m: float(1 - gray[m].mean() / max(med, 1))) for c in _components(dark, int(0.004 * disk.sum()))),
                    key=lambda d: -d["area"])
        stats["n_coronal_holes"] = len(hs)
        stats["coronal_hole_area_pct"] = round(100.0 * sum(d["area"] for d in hs) / float(disk.sum()), 2)
        stats["coronal_holes"] = [{k: d[k] for k in ("location", "lat", "lon", "area_msh")} for d in hs[:8]]
        # a hole near the central meridian sends its fast wind toward Earth a few days later
        stats["earth_facing_hole"] = any(abs(d["lon"]) < 20 and abs(d["lat"]) < 40 for d in hs)
        regs = regs + hs
    return bgr, regs, stats


def annotate(bgr, regs, out_path):
    img = bgr.copy()
    colors = {"sunspot": (0, 255, 255), "active region": (0, 255, 0), "coronal hole": (255, 160, 60)}
    for i, d in enumerate(regs[:60], 1):
        color = colors.get(d["label"], (255, 255, 255))
        pad = 6
        x0, y0 = int(d["x"] - d["a"] - pad), int(d["y"] - d["b"] - pad)
        x1, y1 = int(d["x"] + d["a"] + pad), int(d["y"] + d["b"] + pad)
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2, cv2.LINE_AA)
        cv2.putText(img, d.get("location") or str(i), (x0, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return out_path
