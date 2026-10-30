"""Context for an alert: its light curve, and what catalogues already know about that spot.

* Light curve from ALeRCE (ZTF magnitudes per filter; Rubin difference fluxes converted to AB magnitudes)
* SIMBAD: names and object types of known sources within a few arcseconds
* Gaia DR3 (ESA archive): parallax -> distance, so a "transient" sitting on a nearby star is caught
* NED: redshift of a host galaxy -> light-travel distance (Planck 2018 cosmology)

SIMBAD is queried through its Harvard (CfA) mirror: the Strasbourg servers are not reachable from every network.

Results are cached per object (light curve: 30 min; catalogues: forever, the sky does not move).
"""
import json
import logging
import math
import time
import warnings

import numpy as np

from .config import DATA

log = logging.getLogger("cosmic_watch")
CACHE = DATA / "context"
LC_TTL = 30 * 60
RADIUS_ARCSEC = 5.0
ZTF_BANDS = {1: "g", 2: "r", 3: "i"}
SIMBAD_TAP = "https://simbad.cfa.harvard.edu/simbad/sim-tap"
GAIA_TAP = "https://gea.esac.esa.int/tap-server/tap/sync"


def _cached(name, ttl, fn):
    """ttl None = keep forever, but only results marked complete (a failed service is retried next time)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"{name}.json"
    if p.exists() and (ttl is None or time.time() - p.stat().st_mtime < ttl):
        return json.loads(p.read_text(encoding="utf-8"))
    out = fn()
    if ttl is not None or out.get("complete", True):
        p.write_text(json.dumps(out), encoding="utf-8")
    return out


# ---------- light curve ----------

def light_curve(survey, oid):
    def get():
        from .fetchers.alerts import client
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = client().query_detections(int(oid) if survey == "lsst" else oid, survey=survey, format="pandas")
        pts = []
        if df is None or len(df) == 0:
            return {"points": []}
        for r in df.to_dict("records"):
            if survey == "ztf":
                if r.get("magpsf") is None or (isinstance(r["magpsf"], float) and math.isnan(r["magpsf"])):
                    continue
                pts.append({"mjd": r["mjd"], "band": ZTF_BANDS.get(int(r.get("fid") or 0), "?"), "mag": round(float(r["magpsf"]), 3),
                            "err": round(float(r.get("sigmapsf") or 0), 3), "positive": int(r.get("isdiffpos") or 1) > 0})
            else:
                f, fe = r.get("psfFlux"), r.get("psfFluxErr")
                if f is None or not np.isfinite(f) or f <= 0:
                    continue
                pts.append({"mjd": r["mjd"], "band": r.get("band_name") or str(r.get("band")),
                            "mag": round(31.4 - 2.5 * math.log10(f), 3),             # nJy -> AB mag
                            "err": round(1.0857 * (fe or 0) / f, 3), "positive": True})
        pts.sort(key=lambda p: p["mjd"])
        mags = [p["mag"] for p in pts]
        out = {"points": pts, "n": len(pts)}
        if len(pts) >= 2:
            span = pts[-1]["mjd"] - pts[0]["mjd"]
            out.update(span_days=round(span, 2), brightest=min(mags), faintest=max(mags),
                       amplitude=round(max(mags) - min(mags), 2),
                       trend=round((pts[-1]["mag"] - pts[0]["mag"]) / max(span, 1e-3), 3))   # mag/day (+ = fading)
        return out
    return _cached(f"lc_{survey}_{oid}", LC_TTL, get)


# ---------- catalogue cross-match ----------

def _simbad(ra, dec):
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from astroquery.simbad import Simbad
    s = Simbad()
    s.server = "simbad.harvard.edu"
    s.add_votable_fields("otype", "rvz_redshift", "plx_value")
    c = SkyCoord(ra, dec, unit="deg")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = s.query_region(c, radius=RADIUS_ARCSEC * u.arcsec)
    out = []
    if t is None:
        return out
    cols = {k.lower(): k for k in t.colnames}
    for r in t:
        rr, dd = float(r[cols["ra"]]), float(r[cols["dec"]])
        sep = SkyCoord(rr, dd, unit="deg").separation(c).arcsec
        z = r[cols["rvz_redshift"]] if "rvz_redshift" in cols else None
        out.append({"name": str(r[cols["main_id"]]), "type": str(r[cols["otype"]]) if "otype" in cols else None,
                    "sep_arcsec": round(float(sep), 2),
                    "redshift": float(z) if z is not None and not np.ma.is_masked(z) else None})
    out.sort(key=lambda x: x["sep_arcsec"])
    return out[:5]


def _gaia(ra, dec):
    import requests
    r = RADIUS_ARCSEC / 3600
    q = (f"SELECT TOP 5 source_id, phot_g_mean_mag, parallax, parallax_error, "
         f"DISTANCE(POINT(ra, dec), POINT({ra}, {dec})) * 3600 AS sep FROM gaiadr3.gaia_source "
         f"WHERE 1 = CONTAINS(POINT(ra, dec), CIRCLE({ra}, {dec}, {r})) ORDER BY sep")
    resp = requests.get(GAIA_TAP, params={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "json", "QUERY": q}, timeout=90)
    resp.raise_for_status()
    j = resp.json()
    names = [c["name"] for c in j["metadata"]]
    out = []
    for row in j["data"]:
        d = dict(zip(names, row))
        plx, eplx = d.get("parallax"), d.get("parallax_error")
        good = plx is not None and eplx and plx > 3 * eplx
        out.append({"source_id": str(d["source_id"]), "sep_arcsec": round(float(d["sep"]), 2),
                    "gmag": round(float(d["phot_g_mean_mag"]), 2) if d.get("phot_g_mean_mag") is not None else None,
                    "parallax_mas": round(float(plx), 3) if plx is not None else None,
                    "distance_ly": round(3261.56 / float(plx)) if good else None})
    return out


def _ned(ra, dec):
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from astroquery.ipac.ned import Ned
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = Ned.query_region(SkyCoord(ra, dec, unit="deg"), radius=RADIUS_ARCSEC * 2 * u.arcsec)
    out = []
    if t is None:
        return out
    for r in t:
        z = r["Redshift"]
        out.append({"name": str(r["Object Name"]), "type": str(r["Type"]), "sep_arcsec": round(float(r["Separation"]) * 60, 2),
                    "redshift": float(z) if z is not None and not np.ma.is_masked(z) else None})
    out.sort(key=lambda x: (x["redshift"] is None, x["sep_arcsec"]))
    return out[:5]


def _distance_from_z(z):
    """Light-travel distance in light-years (Planck 2018): how long the light has been on its way."""
    if z is None or z <= 0:
        return None
    from astropy.cosmology import Planck18
    return round(Planck18.lookback_time(z).to_value("yr"))


def crossmatch(ra, dec):
    def get():
        out = {"radius_arcsec": RADIUS_ARCSEC}
        for key, fn in (("simbad", _simbad), ("gaia", _gaia), ("ned", _ned)):
            try:
                out[key] = fn(ra, dec)
            except Exception as e:  # noqa: BLE001
                log.warning("cross-match %s failed: %s", key, e)
                out[key] = None
        out["complete"] = all(out.get(k) is not None for k in ("simbad", "gaia", "ned"))
        # a redshift means distance only for galaxies/quasars (stars have radial velocities, not cosmological redshift)
        z = next((m["redshift"] for m in (out.get("ned") or []) + (out.get("simbad") or [])
                  if m.get("redshift") and m["redshift"] >= 0.002 and "*" not in (m.get("type") or "")), None)
        star = next((g for g in (out.get("gaia") or []) if g.get("distance_ly") and g["sep_arcsec"] < 1.0), None)
        out["summary"] = {"redshift": z, "distance_ly": _distance_from_z(z),
                          "on_milky_way_star": bool(star), "star_distance_ly": star["distance_ly"] if star else None,
                          "known_name": (out.get("simbad") or [{}])[0].get("name") if out.get("simbad") else None,
                          "known_type": (out.get("simbad") or [{}])[0].get("type") if out.get("simbad") else None}
        return out
    return _cached(f"xm_{ra:.5f}_{dec:.5f}", None, get)
