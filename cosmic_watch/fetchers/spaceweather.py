"""Space weather, live from NOAA SWPC and SOHO (all public, no keys).

* GOES X-ray flux (1-minute, 0.1-0.8 nm): the official solar-flare signal, A/B/C/M/X classes
* GOES flare list (7 days), NOAA active regions with C/M/X flare probabilities
* Planetary Kp index (geomagnetic activity -> aurora), real-time solar wind (speed, density, Bz)
* SOHO LASCO C2 / C3 coronagraphs: the Sun's corona out to 30 solar radii, where CMEs show up.
  Our CV runs a running-difference CME detector on consecutive frames.
"""
import hashlib
import logging
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import cv2
import numpy as np
import requests

from ..config import IMG_DIR

log = logging.getLogger("cosmic_watch")
SWPC = "https://services.swpc.noaa.gov"
LASCO = "https://soho.nascom.nasa.gov/data/realtime/{cam}/1024/latest.jpg"
LASCO_KEEP = 16          # frames kept per camera (running difference + a short movie)
_http = requests.Session()
_http.headers["User-Agent"] = "cosmic-watch-ai (github.com/jay4linkh/cosmic-watch-ai)"


def _get(path):
    r = _http.get(SWPC + path, timeout=40)
    r.raise_for_status()
    return r.json()


def flare_class(flux):
    """GOES 0.1-0.8 nm flux (W/m²) -> 'C3.2' style class."""
    if not flux or flux <= 0:
        return None
    for letter, base in (("X", 1e-4), ("M", 1e-5), ("C", 1e-6), ("B", 1e-7), ("A", 1e-8)):
        if flux >= base:
            return f"{letter}{flux / base:.1f}"
    return "A0.0"


def _xrays():
    rows = [r for r in _get("/json/goes/primary/xrays-1-day.json") if r.get("energy") == "0.1-0.8nm" and r.get("flux")]
    series = [[r["time_tag"], r["flux"]] for r in rows[::5]]            # 5-minute cadence for the chart
    now = rows[-1] if rows else None
    peak = max(rows, key=lambda r: r["flux"]) if rows else None
    return {"series": series, "now": now and {"time": now["time_tag"], "flux": now["flux"], "class": flare_class(now["flux"])},
            "peak_24h": peak and {"time": peak["time_tag"], "flux": peak["flux"], "class": flare_class(peak["flux"])}}


def _flares():
    out = []
    for f in _get("/json/goes/primary/xray-flares-7-day.json"):
        if f.get("max_class"):
            out.append({"begin": f.get("begin_time"), "max": f.get("max_time"), "end": f.get("end_time"),
                        "class": f["max_class"], "flux": f.get("max_xrlong")})
    out.sort(key=lambda f: f["max"] or "", reverse=True)
    return out


def _kp():
    one = _get("/json/planetary_k_index_1m.json")
    three = _get("/products/noaa-planetary-k-index.json")
    series = [[r["time_tag"], r.get("Kp")] for r in three if isinstance(r, dict)][-24:]
    now = one[-1] if one else None
    return {"now": now and {"time": now["time_tag"], "kp": now.get("estimated_kp"), "label": now.get("kp")}, "series": series}


def _wind():
    def pick(rows, keys):
        act = [r for r in rows if r.get("active")] or rows
        act = [r for r in act if all(r.get(k) is not None for k in keys)]
        return sorted(act, key=lambda r: r["time_tag"])        # the feed is newest-first
    w = pick(_get("/json/rtsw/rtsw_wind_1m.json"), ["proton_speed", "proton_density"])
    m = pick(_get("/json/rtsw/rtsw_mag_1m.json"), ["bz_gsm", "bt"])
    return {
        "now": {"time": w[-1]["time_tag"] if w else None, "speed": w[-1]["proton_speed"] if w else None,
                "density": w[-1]["proton_density"] if w else None, "temperature": w[-1].get("proton_temperature") if w else None,
                "bz": m[-1]["bz_gsm"] if m else None, "bt": m[-1]["bt"] if m else None, "source": w[-1].get("source") if w else None},
        "speed_series": [[r["time_tag"], r["proton_speed"]] for r in w[-360::10]],
        "bz_series": [[r["time_tag"], r["bz_gsm"]] for r in m[-360::10]],
    }


def _regions():
    rows = _get("/json/solar_regions.json")
    if not rows:
        return {"date": None, "regions": []}
    latest = max(r["observed_date"] for r in rows)
    regs = [{"region": r["region"], "location": r["location"], "area": r["area"], "spots": r["number_spots"],
             "spot_class": r["spot_class"], "mag_class": r["mag_class"],
             "p_c": r.get("c_flare_probability"), "p_m": r.get("m_flare_probability"), "p_x": r.get("x_flare_probability")}
            for r in rows if r["observed_date"] == latest]
    regs.sort(key=lambda r: -(r["p_m"] or 0) * 100 - (r["area"] or 0) / 100)
    probs = _get("/json/solar_probabilities.json")
    return {"date": latest, "regions": regs, "probabilities": probs[0] if probs else None}


# ---------- LASCO + CME detection ----------

def _lasco_mask(shape, cam):
    """Usable ring: outside the occulter, inside the vignetted edge, minus the timestamp box."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(xx - w / 2, yy - h / 2) / w
    inner, outer = (0.215, 0.46) if cam == "c2" else (0.075, 0.46)
    m = (r > inner) & (r < outer)
    m[int(h * 0.94):, : int(w * 0.32)] = False
    return m


def cme_check(prev_path, cur_path, cam):
    """Running difference between two frames: a CME is a large bright front that appears between them.

    Returns {"score", "area_frac", "angle_deg", "bbox"} for the biggest brightening, or None."""
    a = cv2.imread(str(prev_path), cv2.IMREAD_GRAYSCALE)
    b = cv2.imread(str(cur_path), cv2.IMREAD_GRAYSCALE)
    if a is None or b is None or a.shape != b.shape:
        return None
    m = _lasco_mask(a.shape, cam)
    a = cv2.GaussianBlur(a.astype(np.float32), (0, 0), 3)
    b = cv2.GaussianBlur(b.astype(np.float32), (0, 0), 3)
    b *= np.median(a[m]) / max(np.median(b[m]), 1)                     # exposure differences
    rel = (b - a) / (a + 12)
    hot = ((rel > 0.35) & m).astype(np.uint8)
    hot = cv2.morphologyEx(hot, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(hot)
    if n <= 1:
        return {"score": 0.0, "area_frac": 0.0}
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    frac = float(st[k, cv2.CC_STAT_AREA] / m.sum())
    cx, cy = cen[k]
    h, w = a.shape
    angle = float((np.degrees(np.arctan2(-(cx - w / 2), -(cy - h / 2))) + 360) % 360)   # position angle, N = 0, E = 90
    return {"score": round(min(1.0, frac / 0.04), 3), "area_frac": round(frac, 4), "angle_deg": round(angle),
            "bbox": [int(st[k, 0]), int(st[k, 1]), int(st[k, 2]), int(st[k, 3])]}


def _lasco():
    out = {}
    for cam in ("c2", "c3"):
        d = IMG_DIR / "lasco" / cam
        d.mkdir(parents=True, exist_ok=True)
        try:
            r = _http.get(LASCO.format(cam=cam), timeout=60)
            r.raise_for_status()
        except Exception as e:  # noqa: BLE001
            log.warning("LASCO %s: %s", cam, e)
            continue
        lm = r.headers.get("Last-Modified")
        t = parsedate_to_datetime(lm) if lm else datetime.now(timezone.utc)
        path = d / f"{cam}_{t.strftime('%Y%m%d_%H%M%S')}_{hashlib.sha1(r.content).hexdigest()[:6]}.jpg"
        if not path.exists():
            path.write_bytes(r.content)
        frames = sorted(d.glob(f"{cam}_*.jpg"))
        for old in frames[:-LASCO_KEEP]:
            old.unlink(missing_ok=True)
        frames = frames[-LASCO_KEEP:]
        cme = cme_check(frames[-2], frames[-1], cam) if len(frames) >= 2 else None
        out[cam] = {"latest": frames[-1].relative_to(IMG_DIR.parent).as_posix(), "time": t.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "frames": [f.relative_to(IMG_DIR.parent).as_posix() for f in frames], "cme": cme}
    return out


def fetch():
    """One snapshot of everything (each part fails independently)."""
    snap = {}
    for key, fn in (("xrays", _xrays), ("flares", _flares), ("kp", _kp), ("wind", _wind), ("regions", _regions), ("lasco", _lasco)):
        try:
            snap[key] = fn()
        except Exception as e:  # noqa: BLE001
            log.warning("space weather %s failed: %s", key, e)
            snap[key] = None
    if not any(snap.values()):
        raise RuntimeError("all space-weather feeds failed")
    return snap
