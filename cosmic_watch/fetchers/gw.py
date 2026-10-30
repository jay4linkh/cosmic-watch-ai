"""Gravitational waves: real LIGO / Virgo / KAGRA public alerts from GraceDB.

Only "Production" superevents marked significant (SIGNIF_LOCKED) are kept; "MS..." mock-data-challenge
events, which LIGO injects to test its systems, are ignored, as are retracted alerts.
For each event we read the public alert (classification: BBH / BNS / NSBH / terrestrial, has-neutron-star,
has-remnant), the luminosity distance from the sky map's FITS header, and the sky-map picture.
"""
import base64
import io
import logging
import re
from datetime import datetime, timezone

import requests

from ..config import IMG_DIR

log = logging.getLogger("cosmic_watch")
API = "https://gracedb.ligo.org/api/superevents/"
GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)
LEAP = 18                                     # GPS - UTC seconds
_http = requests.Session()
_http.headers["User-Agent"] = "cosmic-watch-ai (github.com/jay4linkh/cosmic-watch-ai)"


def _gps_to_iso(t):
    return datetime.fromtimestamp(GPS_EPOCH.timestamp() + float(t) - LEAP, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _skymap_header(b64):
    """Distance (Mpc) from the multi-order sky map embedded in the alert, without astropy."""
    try:
        raw = base64.b64decode(b64)[:40000]
        txt = raw.decode("latin1")
        out = {}
        for key in ("DISTMEAN", "DISTSTD"):
            m = re.search(rf"{key}\s*=\s*([-+0-9.Ee]+)", txt)
            if m:
                out[key] = float(m.group(1))
        return out
    except Exception:  # noqa: BLE001
        return {}


def _alert_json(sid, files):
    for stage in ("update", "initial", "preliminary"):
        name = f"{sid}-{stage}.json"
        if name in files:
            r = _http.get(files[name], timeout=60)
            if r.ok:
                return stage, r.json()
    return None, None


def dark_skymap(src, dst):
    """LIGO's sky maps are drawn on white: invert lightness (keep hue) so they sit on a dark sky."""
    import cv2
    import numpy as np
    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        return False
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.int16)
    hls[..., 1] = 255 - hls[..., 1]
    cv2.imwrite(str(dst), cv2.cvtColor(hls.astype(np.uint8), cv2.COLOR_HLS2BGR))
    return True


def details(e):
    sid = e["superevent_id"]
    files = _http.get(f"{API}{sid}/files/", timeout=60).json()
    if any(k.endswith("-retraction.json") for k in files):
        return None
    stage, alert = _alert_json(sid, files)
    ev = (alert or {}).get("event") or {}
    cls = ev.get("classification") or {}
    props = ev.get("properties") or {}
    dist = _skymap_header(ev.get("skymap")) if ev.get("skymap") else {}
    img = None
    for name in ("bayestar.png", "Bilby.png", "amplfi.png"):
        if name in files:
            path = IMG_DIR / "gw" / f"{sid}.png"
            if not path.exists():
                r = _http.get(files[name], timeout=60)
                if r.ok:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(r.content)
            dark = path.with_name(f"{sid}_dark.png")
            if path.exists() and (dark.exists() or dark_skymap(path, dark)):
                img = dark.relative_to(IMG_DIR.parent).as_posix()
            break
    best = max(cls, key=cls.get) if cls else None
    pref = e.get("preferred_event_data") or {}
    return {
        "id": sid, "time": _gps_to_iso(e["t_0"]), "created": e["created"], "far_hz": e["far"],
        "far_per_year": e["far"] * 3.156e7 if e.get("far") else None,
        "group": pref.get("group"), "pipeline": pref.get("pipeline"), "instruments": pref.get("instruments"),
        "classification": cls, "class": best, "properties": props,
        "distance_mpc": dist.get("DISTMEAN"), "distance_err_mpc": dist.get("DISTSTD"),
        "alert_stage": stage, "skymap": img, "gw_id": e.get("gw_id"),
        "url": f"https://gracedb.ligo.org/superevents/{sid}/view/",
    }


def fetch(known: dict | None = None, n=40):
    """Newest significant real events; `known` (id -> details) avoids re-downloading old ones."""
    known = known or {}
    q = "category: Production SIGNIF_LOCKED"
    r = _http.get(API, params={"format": "json", "count": n, "orderby": "-created", "query": q}, timeout=60)
    r.raise_for_status()
    body = r.json()
    out = []
    for e in body["superevents"]:
        sid = e["superevent_id"]
        if not sid.startswith("S") or "RETRACTED" in e.get("labels", []) or "ADVNO" in e.get("labels", []):
            continue
        if sid in known:
            out.append(known[sid])
            continue
        try:
            d = details(e)
        except Exception as ex:  # noqa: BLE001
            log.warning("GW %s: %s", sid, ex)
            continue
        if d:
            out.append(d)
    return {"events": out, "total_significant": body.get("numRows"),
            "latest": out[0]["time"] if out else None}
