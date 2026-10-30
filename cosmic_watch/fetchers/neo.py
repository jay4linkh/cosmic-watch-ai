"""Near-Earth objects from NASA/JPL (public, no key).

* Close approaches (SBDB CAD API): asteroids passing within 10 lunar distances, last 7 days + next 60
* Sentry: objects with a non-zero (usually tiny) chance of hitting Earth in the next 100 years
* Fireballs: bright meteors detected by US government sensors, with their impact energy
"""
import math
from datetime import datetime, timedelta, timezone

import requests

JPL = "https://ssd-api.jpl.nasa.gov"
AU_KM = 149_597_870.7
LD_KM = 384_400
_http = requests.Session()
_http.headers["User-Agent"] = "cosmic-watch-ai (github.com/jay4linkh/cosmic-watch-ai)"


def size_from_h(h):
    """Diameter range (m) from absolute magnitude H, for albedo 0.25 (bright) to 0.05 (dark)."""
    if h is None:
        return None
    d = lambda p: 1329 / math.sqrt(p) * 10 ** (-h / 5) * 1000
    return [round(d(0.25)), round(d(0.05))]


def _rows(j):
    return [dict(zip(j["fields"], r)) for r in j.get("data", [])]


def close_approaches():
    week_ago = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")
    j = _http.get(f"{JPL}/cad.api", params={"dist-max": "10LD", "date-min": week_ago, "date-max": "+60", "sort": "date",
                                             "fullname": "true", "diameter": "true"}, timeout=60).json()
    out = []
    for r in _rows(j):
        h = float(r["h"]) if r.get("h") else None
        dist_km = float(r["dist"]) * AU_KM
        diam = float(r["diameter"]) * 1000 if r.get("diameter") else None
        out.append({"name": r["fullname"].strip(), "des": r["des"], "time": r["cd"], "jd": float(r["jd"]),
                    "dist_ld": round(dist_km / LD_KM, 3), "dist_km": round(dist_km),
                    "v_kms": round(float(r["v_rel"]), 2), "h": h,
                    "size_m": [round(diam), round(diam)] if diam else size_from_h(h), "size_measured": diam is not None})
    return out


def sentry(n=25):
    j = _http.get(f"{JPL}/sentry.api", timeout=60).json()
    rows = j.get("data", [])
    rows.sort(key=lambda r: -float(r.get("ps_cum") or -99))
    return {"count": int(j.get("count", len(rows))),
            "top": [{"name": r["fullname"].strip(), "des": r["des"], "years": r["range"], "p_impact": float(r["ip"]),
                     "palermo": float(r["ps_cum"]), "torino": int(r.get("ts_max") or 0),
                     "size_m": round(float(r["diameter"]) * 1000) if r.get("diameter") else None,
                     "v_kms": float(r["v_inf"]) if r.get("v_inf") else None, "n_impacts": r.get("n_imp")}
                    for r in rows[:n]]}


def fireballs(n=25):
    j = _http.get(f"{JPL}/fireball.api", params={"limit": n}, timeout=60).json()
    out = []
    for r in _rows(j):
        lat = float(r["lat"]) * (1 if r.get("lat-dir") == "N" else -1) if r.get("lat") else None
        lon = float(r["lon"]) * (1 if r.get("lon-dir") == "E" else -1) if r.get("lon") else None
        out.append({"time": r["date"], "lat": lat, "lon": lon, "alt_km": float(r["alt"]) if r.get("alt") else None,
                    "v_kms": float(r["vel"]) if r.get("vel") else None,
                    "energy_kt": float(r["impact-e"]) if r.get("impact-e") else None})
    return out


def fetch():
    return {"approaches": close_approaches(), "sentry": sentry(), "fireballs": fireballs()}


def iss():
    j = _http.get("https://api.wheretheiss.at/v1/satellites/25544", timeout=15).json()
    return {"lat": j["latitude"], "lon": j["longitude"], "alt_km": j["altitude"], "v_kmh": j["velocity"],
            "visibility": j.get("visibility"), "time": j["timestamp"]}
