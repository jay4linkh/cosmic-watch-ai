"""Catalogue adapters: one cone query per catalogue, normalised sources, provenance, cached.

Every adapter returns
    {"catalog", "ok", "error", "field_count", "field_radius_arcsec", "sources": [...], "provenance": {...}}
and every source is normalised to
    {id, ra, dec, err (arcsec), epoch (decimal yr), pmra, pmdec, pmra_err, pmdec_err, parallax, parallax_err,
     mags: {band: [mag, err]}, morph ("point" | "extended" | None), redshift, redshift_type, name, otype, extra}

All services are public and need no key. SIMBAD goes through its Harvard mirror (Strasbourg is not reachable from
every network). Results are cached on disk: catalogues do not change, so a position is only queried once
(ALeRCE, which grows nightly, is refreshed after a day).
"""
import json
import logging
import math
import time
import warnings
from pathlib import Path

import requests

from ..config import DATA

log = logging.getLogger("cosmic_watch")
CACHE = DATA / "xsurvey" / "cache"
FIELD_RADIUS = 60.0                     # arcsec: cone used for candidates + local source density
TTL = {"alerce_ztf": 86400}
_http = requests.Session()
_http.headers["User-Agent"] = "cosmic-watch-ai (github.com/jay4linkh/cosmic-watch-ai)"

MJD0_YEAR = lambda mjd: 2000.0 + (mjd - 51544.5) / 365.25          # noqa: E731


def _cached(catalog, ra, dec, fn, radius=None):
    tag = f"_r{radius:.0f}" if radius and radius != FIELD_RADIUS else ""
    p = CACHE / catalog / f"{ra:.5f}_{dec:.5f}{tag}.json"
    ttl = TTL.get(catalog)
    if p.exists() and (ttl is None or time.time() - p.stat().st_mtime < ttl):
        return json.loads(p.read_text(encoding="utf-8"))
    t0 = time.time()
    try:
        out = fn()
        out.setdefault("ok", True)
    except Exception as e:  # noqa: BLE001
        log.warning("%s query failed at %.4f %.4f: %s", catalog, ra, dec, e)
        out = {"ok": False, "error": str(e)[:240], "sources": [], "field_count": None}
    out.update(catalog=catalog, field_radius_arcsec=radius or FIELD_RADIUS, latency_s=round(time.time() - t0, 2),
               fetched_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    if out["ok"]:                                  # failures are retried next time, never cached
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out), encoding="utf-8")
    return out


def _box(ra, dec, r_arcsec):
    d = r_arcsec / 3600
    dr = d / max(math.cos(math.radians(dec)), 1e-3)
    return ra - dr, ra + dr, dec - d, dec + d


def _within(src, ra, dec, r_arcsec):
    from .match import sep_arcsec
    return [s for s in src if sep_arcsec(ra, dec, s["ra"], s["dec"]) <= r_arcsec]


def _num(v, bad=(-999, -999.0)):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if (f in bad or math.isnan(f)) else f


# ---------- Gaia DR3 (ESA archive) ----------

GAIA_TAP = "https://gea.esac.esa.int/tap-server/tap/sync"


def gaia(ra, dec, radius=FIELD_RADIUS):
    def q():
        r = radius / 3600
        adql = (f"SELECT TOP {500 if radius <= FIELD_RADIUS else 20000} source_id, ra, dec, ra_error, dec_error, ref_epoch, pmra, pmdec, pmra_error, pmdec_error, "
                f"parallax, parallax_error, phot_g_mean_mag, bp_rp, classprob_dsc_combmod_galaxy, classprob_dsc_combmod_quasar "
                f"FROM gaiadr3.gaia_source WHERE 1 = CONTAINS(POINT(ra, dec), CIRCLE({ra}, {dec}, {r}))")
        resp = _http.get(GAIA_TAP, params={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "json", "QUERY": adql}, timeout=120)
        resp.raise_for_status()
        j = resp.json()
        names = [c["name"] for c in j["metadata"]]
        src = []
        for row in j["data"]:
            d = dict(zip(names, row))
            plx, eplx = _num(d["parallax"]), _num(d["parallax_error"])
            pgal, pqso = _num(d.get("classprob_dsc_combmod_galaxy")), _num(d.get("classprob_dsc_combmod_quasar"))
            src.append({"id": str(d["source_id"]), "ra": d["ra"], "dec": d["dec"],
                        "err": math.hypot(_num(d["ra_error"]) or 0.5, _num(d["dec_error"]) or 0.5) / 1000 / math.sqrt(2),
                        "epoch": _num(d["ref_epoch"]) or 2016.0, "pmra": _num(d["pmra"]), "pmdec": _num(d["pmdec"]),
                        "pmra_err": _num(d["pmra_error"]), "pmdec_err": _num(d["pmdec_error"]),
                        "parallax": plx, "parallax_err": eplx,
                        "mags": {"G": [_num(d["phot_g_mean_mag"]), 0.01]} if _num(d["phot_g_mean_mag"]) else {},
                        "bp_rp": _num(d["bp_rp"]),
                        "morph": "point" if (pgal or 0) < 0.5 else "extended",
                        "extra": {"p_galaxy": pgal, "p_quasar": pqso}})
        return {"sources": src, "field_count": len(src),
                "provenance": {"service": "ESA Gaia archive TAP", "table": "gaiadr3.gaia_source", "query": adql}}
    return _cached("gaia_dr3", ra, dec, q, radius)


# ---------- Pan-STARRS1 DR2 mean catalogue (MAST) ----------

PS1 = "https://catalogs.mast.stsci.edu/api/v0.1/panstarrs/dr2/mean.json"
PS1_COLS = ["objID", "raMean", "decMean", "raMeanErr", "decMeanErr", "epochMean", "nDetections",
            "gMeanPSFMag", "gMeanPSFMagErr", "rMeanPSFMag", "rMeanPSFMagErr", "iMeanPSFMag", "iMeanPSFMagErr", "rMeanKronMag"]


def ps1(ra, dec, radius=FIELD_RADIUS):
    def q():
        if dec < -31:
            return {"sources": [], "field_count": 0, "provenance": {"note": "outside the Pan-STARRS footprint (dec < -30)"}}
        params = {"ra": ra, "dec": dec, "radius": radius / 3600, "nDetections.gte": 1, "pagesize": 500 if radius <= FIELD_RADIUS else 20000,
                  "columns": json.dumps(PS1_COLS)}
        resp = _http.get(PS1, params=params, timeout=120)
        resp.raise_for_status()
        j = resp.json()
        names = [c["name"] for c in j["info"]]
        src = []
        for row in j["data"]:
            d = dict(zip(names, row)) if isinstance(row, list) else row
            if _num(d["raMean"]) is None:
                continue
            n = max(int(d.get("nDetections") or 1), 1)
            e = _num(d["raMeanErr"])
            err = min(max((e * 3600 / math.sqrt(n)) if e else 0.1, 0.02), 1.0)
            psf, kron = _num(d["rMeanPSFMag"]), _num(d["rMeanKronMag"])
            mags = {b: [_num(d[f"{b}MeanPSFMag"]), _num(d[f"{b}MeanPSFMagErr"])] for b in "gri" if _num(d[f"{b}MeanPSFMag"])}
            src.append({"id": str(d["objID"]), "ra": d["raMean"], "dec": d["decMean"], "err": err,
                        "epoch": MJD0_YEAR(_num(d["epochMean"])) if _num(d["epochMean"]) else 2012.4,
                        "mags": mags, "morph": None if psf is None or kron is None else ("extended" if psf - kron > 0.05 else "point"),
                        "extra": {"n_detections": n, "r_kron": kron}})
        return {"sources": src, "field_count": len(src),
                "provenance": {"service": "MAST catalogs API", "table": "panstarrs/dr2/mean", "params": params}}
    return _cached("ps1_dr2", ra, dec, q, radius)


# ---------- SDSS DR18 (SkyServer) ----------

SDSS = "https://skyserver.sdss.org/dr18/SkyServerWS/SearchTools/SqlSearch"


def sdss(ra, dec, radius=FIELD_RADIUS):
    def q():
        sql = (f"select top {5000 if radius > FIELD_RADIUS else 500} p.objid, p.ra, p.dec, p.raErr, p.decErr, p.mjd, p.type, p.g, p.err_g, p.r, p.err_r, "
               f"s.z, s.zErr, s.class from fGetNearbyObjEq({ra},{dec},{radius / 60}) n "
               f"join PhotoObj p on p.objid = n.objid left join SpecObj s on s.bestobjid = p.objid")
        resp = _http.get(SDSS, params={"cmd": sql, "format": "json"}, timeout=120)
        resp.raise_for_status()
        rows = resp.json()[0]["Rows"]
        src = []
        for d in rows:
            mags = {b: [_num(d[b]), _num(d[f"err_{b}"])] for b in "gr" if _num(d[b])}
            src.append({"id": str(d["objid"]), "ra": d["ra"], "dec": d["dec"],
                        "err": max(math.hypot(_num(d["raErr"]) or 0.1, _num(d["decErr"]) or 0.1) / math.sqrt(2), 0.05),
                        "epoch": MJD0_YEAR(d["mjd"]) if d.get("mjd") else 2004.0, "mags": mags,
                        "morph": {3: "extended", 6: "point"}.get(d["type"]),
                        "redshift": _num(d.get("z")), "redshift_type": "spec" if _num(d.get("z")) is not None else None,
                        "otype": d.get("class") or None})
        return {"sources": src, "field_count": len(src),
                "provenance": {"service": "SDSS SkyServer DR18", "query": sql}}
    return _cached("sdss_dr18", ra, dec, q, radius)


# ---------- DESI Legacy Surveys DR10 (NOIRLab Astro Data Lab) ----------

DATALAB = "https://datalab.noirlab.edu/tap/sync"


def legacy(ra, dec, radius=FIELD_RADIUS):
    def q():
        r0, r1, d0, d1 = _box(ra, dec, radius)
        adql = (f"SELECT TOP {800 if radius <= FIELD_RADIUS else 20000} t.ls_id, t.ra, t.dec, t.ra_ivar, t.dec_ivar, t.type, t.shape_r, t.mag_g, t.mag_r, t.mag_z, "
                f"t.flux_r, t.flux_ivar_r, p.z_phot_median, p.z_phot_std, p.z_spec "
                f"FROM ls_dr10.tractor t LEFT JOIN ls_dr10.photo_z p ON t.ls_id = p.ls_id "
                f"WHERE t.ra BETWEEN {r0} AND {r1} AND t.dec BETWEEN {d0} AND {d1}")
        resp = _http.get(DATALAB, params={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": adql}, timeout=120)
        resp.raise_for_status()
        import csv
        import io
        rows = list(csv.DictReader(io.StringIO(resp.text)))
        if rows and "ls_id" not in rows[0]:
            raise RuntimeError(resp.text[:200])
        src = []
        for d in rows:
            rv, dv = _num(d["ra_ivar"]), _num(d["dec_ivar"])
            err = math.sqrt(0.5 / rv + 0.5 / dv) * 3600 if rv and dv else 0.1        # ivar in 1/deg^2
            fr, fi = _num(d["flux_r"]), _num(d["flux_ivar_r"])
            mag_r = _num(d["mag_r"])
            zs, zp = _num(d.get("z_spec"), bad=(-99, -99.0)), _num(d.get("z_phot_median"), bad=(-99, -99.0))
            src.append({"id": d["ls_id"], "ra": float(d["ra"]), "dec": float(d["dec"]), "err": min(max(err, 0.02), 1.0),
                        "epoch": 2017.5,
                        "mags": {b: [_num(d[f"mag_{b}"]), (1.0857 / (fr * math.sqrt(fi)) if b == "r" and fr and fi and fr > 0 else None)]
                                 for b in "grz" if _num(d[f"mag_{b}"])},
                        "morph": "point" if d["type"] == "PSF" else ("extended" if d["type"] not in ("DUP", "") else None),
                        "redshift": zs if zs is not None else zp,
                        "redshift_type": "spec" if zs is not None else ("phot" if zp is not None else None),
                        "extra": {"type": d["type"], "shape_r": _num(d.get("shape_r")), "z_phot_std": _num(d.get("z_phot_std"), bad=(-99, -99.0))}})
        src = _within(src, ra, dec, radius)
        return {"sources": src, "field_count": len(src),
                "provenance": {"service": "NOIRLab Astro Data Lab TAP", "table": "ls_dr10.tractor + ls_dr10.photo_z", "query": adql}}
    return _cached("ls_dr10", ra, dec, q, radius)


# ---------- SIMBAD (CfA mirror) and NED: names, types, redshifts ----------

def simbad(ra, dec, radius=30.0):
    def q():
        from astropy import units as u
        from astropy.coordinates import SkyCoord
        from astroquery.simbad import Simbad
        s = Simbad()
        s.server = "simbad.harvard.edu"
        s.add_votable_fields("otype", "rvz_redshift", "plx_value")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t = s.query_region(SkyCoord(ra, dec, unit="deg"), radius=radius * u.arcsec)
        src = []
        if t is not None:
            cols = {k.lower(): k for k in t.colnames}
            for r in t:
                z = r[cols["rvz_redshift"]] if "rvz_redshift" in cols else None
                z = _num(z) if z is not None and str(z) != "--" else None
                otype = str(r[cols["otype"]]) if "otype" in cols else None
                src.append({"id": str(r[cols["main_id"]]), "name": str(r[cols["main_id"]]), "ra": float(r[cols["ra"]]),
                            "dec": float(r[cols["dec"]]), "err": 0.5, "otype": otype,
                            "redshift": z if z is not None and z >= 0.002 and "*" not in (otype or "") else None,
                            "redshift_type": "simbad", "morph": "point" if "*" in (otype or "") else None, "mags": {}})
        return {"sources": src, "field_count": len(src), "provenance": {"service": "SIMBAD (CfA mirror)", "radius_arcsec": radius}}
    return _cached("simbad", ra, dec, q, radius if radius > FIELD_RADIUS else None)


def ned(ra, dec, radius=30.0):
    def q():
        from astropy import units as u
        from astropy.coordinates import SkyCoord
        from astroquery.ipac.ned import Ned
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t = Ned.query_region(SkyCoord(ra, dec, unit="deg"), radius=radius * u.arcsec)
        src = []
        if t is not None:
            for r in t:
                z = _num(r["Redshift"]) if str(r["Redshift"]) != "--" else None
                src.append({"id": str(r["Object Name"]), "name": str(r["Object Name"]), "ra": float(r["RA"]), "dec": float(r["DEC"]),
                            "err": 1.0, "otype": str(r["Type"]), "redshift": z, "redshift_type": "ned", "mags": {},
                            "morph": "extended" if str(r["Type"]) in ("G", "GPair", "GTrpl", "GGroup") else None})
        return {"sources": src, "field_count": len(src), "provenance": {"service": "NASA/IPAC NED", "radius_arcsec": radius}}
    return _cached("ned", ra, dec, q, radius if radius > FIELD_RADIUS else None)


# ---------- ZTF objects at this position (ALeRCE) ----------

def alerce_ztf(ra, dec, radius=5.0):
    def q():
        resp = _http.get("https://api.alerce.online/ztf/v1/objects/", params={"ra": ra, "dec": dec, "radius": radius,
                                                                               "page_size": 20}, timeout=60)
        resp.raise_for_status()
        src = []
        for it in resp.json().get("items", []):
            src.append({"id": it["oid"], "ra": it["meanra"], "dec": it["meandec"], "err": 0.3,
                        "epoch": MJD0_YEAR(it["firstmjd"]) if it.get("firstmjd") else None, "mags": {},
                        "extra": {"ndet": it.get("ndet"), "firstmjd": it.get("firstmjd"), "lastmjd": it.get("lastmjd"),
                                  "ncovhist": it.get("ncovhist"), "stellar": it.get("stellar")}})
        return {"sources": src, "field_count": None, "provenance": {"service": "ALeRCE ZTF API", "radius_arcsec": radius}}
    return _cached("alerce_ztf", ra, dec, q)


CATALOGS = {"gaia_dr3": gaia, "ps1_dr2": ps1, "sdss_dr18": sdss, "ls_dr10": legacy, "simbad": simbad, "ned": ned,
            "alerce_ztf": alerce_ztf}
LABELS = {"gaia_dr3": "Gaia DR3", "ps1_dr2": "Pan-STARRS1 DR2", "sdss_dr18": "SDSS DR18", "ls_dr10": "Legacy Surveys DR10",
          "simbad": "SIMBAD", "ned": "NED", "alerce_ztf": "ZTF (ALeRCE)", "jpl_sbident": "JPL Small-Body Identification"}


def query_all(ra, dec, only=None, radius=None):
    """All catalogues for one position, in parallel threads (each is network-bound).
    radius: a larger cone for a whole image (one query per field instead of one per detection)."""
    from concurrent.futures import ThreadPoolExecutor
    names = only or list(CATALOGS)
    kw = {"radius": radius} if radius else {}
    with ThreadPoolExecutor(max_workers=len(names)) as ex:
        futs = {n: ex.submit(CATALOGS[n], ra, dec, **({} if n == "alerce_ztf" else kw)) for n in names}
        return {n: f.result() for n, f in futs.items()}


# ---------- known Solar System bodies at this place and time (JPL) ----------

MPC_CODE = {"lsst": "X05", "ztf": "I41"}


def jpl_sbident(ra, dec, mjd, survey="lsst", half_width_deg=0.01):
    """Asteroids / comets predicted within a small box around (ra, dec) at the alert time (JPL two-pass ephemerides)."""
    def q():
        from astropy import units as u
        from astropy.coordinates import Angle
        from astropy.time import Time
        params = {"mpc-code": MPC_CODE.get(survey, "500"), "obs-time": Time(mjd, format="mjd").strftime("%Y-%m-%d_%H:%M:%S"),
                  "fov-ra-center": str(Angle(ra, u.deg).to_string(unit=u.hour, sep="-", precision=2, pad=True)),
                  "fov-dec-center": str(Angle(dec, u.deg).to_string(sep="-", precision=1, alwayssign=True, pad=True)),
                  "fov-ra-hwidth": half_width_deg, "fov-dec-hwidth": half_width_deg, "two-pass": "true", "suppress-first-pass": "true"}
        resp = _http.get("https://ssd-api.jpl.nasa.gov/sb_ident.api", params=params, timeout=240)
        resp.raise_for_status()
        j = resp.json()
        if "code" in j and str(j["code"]) != "200":
            raise RuntimeError(j.get("message"))
        names = j.get("fields_second") or []
        src = []
        for row in j.get("data_second_pass") or []:
            d = dict(zip(names, row))
            src.append({"id": d["Object name"], "name": d["Object name"], "sep": float(d['Dist. from center Norm (")']),
                        "V": _num(d.get("Visual magnitude (V)")), "rate_ra": _num(d.get('RA rate ("/h)')),
                        "rate_dec": _num(d.get('Dec rate ("/h)'))})
        return {"sources": src, "field_count": len(src), "provenance": {"service": "NASA/JPL Small-Body Identification API", "params": params}}
    return _cached(f"jpl_sbident/{mjd:.4f}", ra, dec, q)
