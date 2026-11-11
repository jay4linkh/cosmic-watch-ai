"""Vetting rules for anomaly candidates.

Written after an audit of the first 744 candidates (30 Oct - 10 Nov 2026) found three failure modes:
  * deep-field outliers were mostly bright stars and non-galaxies, because "normal" was defined over
    every source, stars included, so the brightest point sources looked the strangest;
  * broker TDE labels were raised for objects ZTF first saw years ago (a tidal disruption fades within
    about a year) and for objects the archives identify as variable stars or AGN;
  * "hostless" transients included single-detection alerts the broker itself classed as asteroids.

The pipeline (anomaly.py, xsurvey/engine.py) and the re-measurement (`python -m cosmic_watch.vetting`)
use these same functions, so the numbers reported are the rules that run.
"""
import json
import re
import time

TDE_MAX_AGE_YR = 1          # a TDE rises and fades within about a year; older ZTF objects are AGN or variables
NEW_MAX_AGE_YR = 1          # "new" means first seen within about a year
ARCHIVE_SAYS_NOT_TDE = {"variable_star": "the archives identify a Milky Way variable star here",
                        "agn_variability": "the archives list a known AGN / quasar here",
                        "known_source_variable": "a catalogued source sits here with no host galaxy around it"}
BROKER_SAYS_AGN = {"AGN", "QSO", "Blazar"}


def _year_now():
    return time.gmtime().tm_year


def first_seen_year(survey, oid):
    """ZTF object names encode the year of first detection (ZTF18... = 2018). Others: unknown (None)."""
    m = re.match(r"ZTF(\d\d)", str(oid or ""))
    return 2000 + int(m.group(1)) if (survey or "").lower() == "ztf" and m else None


def tde_ok(survey, oid, xs_verdict=None):
    """(keep, reason) for a broker TDE label."""
    y = first_seen_year(survey, oid)
    if y is not None and _year_now() - y > TDE_MAX_AGE_YR:
        return False, f"first detected in {y}: too long-lived for a tidal disruption (more likely an AGN or a variable)"
    if xs_verdict in ARCHIVE_SAYS_NOT_TDE:
        return False, ARCHIVE_SAYS_NOT_TDE[xs_verdict]
    return True, ""


def hostless_ok(survey, oid, n_det, broker_cls, moving_ruled_out):
    """(keep, reason) for a 'new, hostless transient' verdict."""
    if "asteroid" in (broker_cls or "").lower():
        return False, "the broker classes it as an asteroid"
    if (n_det or 0) < 2:
        return False, "seen only once: a moving object or an artefact is not ruled out"
    if not moving_ruled_out:
        return False, "no successful check against known asteroids at the alert time"
    y = first_seen_year(survey, oid)
    if y is not None and _year_now() - y > NEW_MAX_AGE_YR:
        return False, f"first detected in {y}, so not new"
    return True, ""


def supernova_ok(broker_cls):
    """(keep, reason) for a host-offset supernova verdict."""
    if (broker_cls or "") in BROKER_SAYS_AGN:
        return False, f"the broker's light-curve classifier says {broker_cls}, not a supernova"
    return True, ""


def moving_ruled_out_from_report(report):
    """True when the stored evidence shows a successful JPL check with no known body nearby."""
    return any(e.get("source") == "jpl_sbident" and e.get("claim_type") == "catalog_fact"
               and str(e.get("statement", "")).startswith("No known asteroid") for e in report.get("evidence", []))


def deep_population(dets):
    """Sources that define 'normal' in a deep field: extended sources the CNN has not rejected."""
    return [d for d in dets if d["label"] != "star" and (d["cnn_label"] or "") != "Not a galaxy"]


# ---------- re-measure / clean up the candidates already raised ----------

def revalidate(con, apply=False):
    """Re-check every open object-level candidate with the rules above.
    Returns {kind: {"before": n, "kept": n, "retracted": n, "reasons": {reason: n}}}; with apply=True the
    failures are marked status='retracted' (the agent only shows status='new')."""
    from collections import Counter, defaultdict
    from . import anomaly
    out = defaultdict(lambda: {"before": 0, "kept": 0, "retracted": 0, "reasons": Counter()})
    retract = []

    def alert(survey, oid):
        return con.execute("SELECT * FROM alerts WHERE survey=? AND oid=? ORDER BY id DESC LIMIT 1", (survey, oid)).fetchone()

    def xs(survey, oid):
        return con.execute("SELECT verdict, report FROM astronomical_objects WHERE ref=?", (f"alert:{survey}:{oid}",)).fetchone()

    # deep-field outliers: recompute per image with the new population, keep those still selected
    deep = con.execute("SELECT id, key, ref FROM anomalies WHERE kind='deep_outlier' AND status!='retracted'").fetchall()
    by_img = defaultdict(list)
    for a in deep:
        by_img[json.loads(a["ref"])["image_id"]].append(a)
    for iid, rows in by_img.items():
        sel = anomaly.deep_outliers(con, iid)          # what the vetted rule selects for this image now
        still = {x["key"] for x in sel}
        out["deep_outlier"].setdefault("newly_selected", 0)
        out["deep_outlier"]["newly_selected"] += sum(x["key"] not in {a["key"] for a in rows} for x in sel)
        for a in rows:
            o = out["deep_outlier"]; o["before"] += 1
            if a["key"] in still:
                o["kept"] += 1
            else:
                det = con.execute("SELECT label, cnn_label FROM detections WHERE id=?", (json.loads(a["ref"]).get("det_id"),)).fetchone()
                why = ("a star" if det and det["label"] == "star" else "rejected by the CNN as not a galaxy"
                       if det and det["cnn_label"] == "Not a galaxy" else "no longer an outlier among extended sources")
                o["retracted"] += 1; o["reasons"][why] += 1; retract.append(a["id"])

    for a in con.execute("SELECT id, kind, ref FROM anomalies WHERE kind IN ('tde_candidate','xs_new_hostless_transient',"
                         "'xs_supernova_candidate') AND status!='retracted'").fetchall():
        r = json.loads(a["ref"]); survey, oid = r.get("survey"), r.get("oid")
        al, x = alert(survey, oid), xs(survey, oid)
        if a["kind"] == "tde_candidate":
            ok, why = tde_ok(survey, oid, x["verdict"] if x else None)
        elif a["kind"] == "xs_new_hostless_transient":
            ok, why = hostless_ok(survey, oid, al["n_det"] if al else 0, al["cls"] if al else None,
                                  moving_ruled_out_from_report(json.loads(x["report"])) if x else False)
        else:
            ok, why = supernova_ok(al["cls"] if al else None)
        o = out[a["kind"]]; o["before"] += 1
        if ok:
            o["kept"] += 1
        else:
            o["retracted"] += 1; o["reasons"][why if "first detected" not in why else "first detected years ago"] += 1
            retract.append(a["id"])
    if apply and retract:
        con.executemany("UPDATE anomalies SET status='retracted' WHERE id=?", [(i,) for i in retract])
        con.commit()
    return {k: {**v, "reasons": dict(v["reasons"])} for k, v in out.items()}


if __name__ == "__main__":
    import argparse
    from . import db
    ap = argparse.ArgumentParser(description="Re-check anomaly candidates with the vetting rules.")
    ap.add_argument("--apply", action="store_true", help="mark failing candidates as retracted")
    args = ap.parse_args()
    con = db.connect()
    print(json.dumps(revalidate(con, apply=args.apply), indent=1))
