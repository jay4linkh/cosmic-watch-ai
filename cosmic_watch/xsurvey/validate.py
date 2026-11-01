"""How good is the matcher? Measured against an independent answer key.

Gaia DR3 ships its own Pan-STARRS1 cross-match (gaiadr3.panstarrs1_best_neighbour, Marrese et al. 2019/2022),
built with a different algorithm (proper motions, local density, mutual-neighbour checks). For random fields we
take the Gaia sources that have a best neighbour, run OUR matcher on PS1 candidates around each Gaia position and
count how often our top candidate (P >= 0.5) is the same PS1 objID.

Reported: precision (of our confident matches, how many are right), recall (of Gaia's matches, how many we find
confidently) and the calibration of P (mean P vs observed accuracy, per bin). Writes data/xsurvey/validation.json
and logs a model_versions row with the metrics.

    python -m cosmic_watch.xsurvey.validate [n_fields]
"""
import json
import logging
import random
import sys
import time

import requests

from ..config import DATA
from . import catalogs
from .match import match, sep_arcsec

log = logging.getLogger("cosmic_watch")
OUT = DATA / "xsurvey" / "validation.json"


def _gaia_with_ps1(ra, dec, r_arcsec=60):
    adql = (f"SELECT g.source_id, g.ra, g.dec, g.ra_error, g.dec_error, g.ref_epoch, g.pmra, g.pmdec, g.pmra_error, g.pmdec_error, "
            f"g.phot_g_mean_mag, n.original_ext_source_id "
            f"FROM gaiadr3.gaia_source g JOIN gaiadr3.panstarrs1_best_neighbour n ON g.source_id = n.source_id "
            f"WHERE 1 = CONTAINS(POINT(g.ra, g.dec), CIRCLE({ra}, {dec}, {r_arcsec / 3600}))")
    r = requests.get(catalogs.GAIA_TAP, params={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "json", "QUERY": adql}, timeout=180)
    r.raise_for_status()
    j = r.json()
    names = [c["name"] for c in j["metadata"]]
    return [dict(zip(names, row)) for row in j["data"]]


def run(n_fields=8, seed=7):
    rng = random.Random(seed)
    fields = []
    while len(fields) < n_fields:                   # random sky positions inside the PS1 footprint, away from the plane
        ra, dec = rng.uniform(0, 360), rng.uniform(-25, 80)
        from astropy import units as u
        from astropy.coordinates import SkyCoord
        b = SkyCoord(ra, dec, unit="deg").galactic.b.deg
        if abs(b) > 15 or len(fields) < n_fields // 4:   # mostly high latitude, a few crowded fields
            fields.append((ra, dec, round(b, 1)))
    tp = fp = fn = 0
    bins = {i: [0, 0, 0.0] for i in range(5)}      # P bin -> [n, correct, sum P]
    per_field = []
    for ra, dec, b in fields:
        t0 = time.time()
        try:
            truth = _gaia_with_ps1(ra, dec)
            ps = catalogs.ps1(ra, dec, radius=90)
        except Exception as e:  # noqa: BLE001
            log.warning("validation field %.2f %.2f skipped: %s", ra, dec, e)
            continue
        if not ps["ok"]:
            continue
        src = ps["sources"]
        f_tp = f_fp = f_fn = 0
        for g in truth:
            if sep_arcsec(ra, dec, g["ra"], g["dec"]) > 60:
                continue
            target = {"ra": g["ra"], "dec": g["dec"], "epoch": g["ref_epoch"],
                      "err": max(g["ra_error"] or 0.1, g["dec_error"] or 0.1) / 1000}
            # PS1 positions are mean epoch ~2012: move the Gaia star back with its proper motion via the target epoch
            from .match import propagate
            if g["pmra"] is not None:
                target["ra"], target["dec"] = propagate(g["ra"], g["dec"], g["pmra"], g["pmdec"], g["ref_epoch"], 2012.0)
            cand = [s for s in src if sep_arcsec(target["ra"], target["dec"], s["ra"], s["dec"]) < 10]
            ms = match(target, cand, field_count=len(src), field_radius_arcsec=90)
            top = ms[0] if ms else None
            want = str(g["original_ext_source_id"])
            if top and top["p_match"] >= 0.5:
                ok = top["id"] == want
                f_tp += ok
                f_fp += not ok
                k = min(int(top["p_match"] * 5), 4)
                bins[k][0] += 1
                bins[k][1] += ok
                bins[k][2] += top["p_match"]
            else:
                f_fn += 1
        tp, fp, fn = tp + f_tp, fp + f_fp, fn + f_fn
        per_field.append({"ra": round(ra, 3), "dec": round(dec, 3), "b": b, "n": f_tp + f_fp + f_fn, "tp": f_tp, "fp": f_fp,
                          "fn": f_fn, "seconds": round(time.time() - t0, 1)})
        log.info("field %.2f %+.2f (b=%.0f): %d true, %d wrong, %d missed", ra, dec, b, f_tp, f_fp, f_fn)
    res = {"answer_key": "gaiadr3.panstarrs1_best_neighbour", "fields": per_field,
           "precision": round(tp / max(tp + fp, 1), 4), "recall": round(tp / max(tp + fn, 1), 4), "n": tp + fp + fn,
           "calibration": [{"p_bin": f"{i / 5:.1f}-{(i + 1) / 5:.1f}", "n": n, "mean_p": round(sp / n, 3), "accuracy": round(c / n, 3)}
                           for i, (n, c, sp) in bins.items() if n],
           "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    try:
        from .. import db
        from .engine import ENGINE_VERSION
        con = db.connect()
        con.execute("""INSERT INTO model_versions (name, version, created_at, metrics, notes) VALUES (?,?,?,?,?)
                       ON CONFLICT(name, version) DO UPDATE SET metrics=excluded.metrics, created_at=excluded.created_at""",
                    ("xsurvey-match", ENGINE_VERSION, res["created_at"],
                     json.dumps({k: res[k] for k in ("precision", "recall", "n")}), "validated vs Gaia DR3 PS1 best neighbours"))
        con.commit()
        con.close()
    except Exception as e:  # noqa: BLE001
        log.warning("could not record validation: %s", e)
    return res


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    r = run(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
    print(json.dumps({k: v for k, v in r.items() if k != "fields"}, indent=1))
