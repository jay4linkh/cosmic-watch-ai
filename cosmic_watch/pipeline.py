"""Fetch -> detect -> store, for every live source.

    python -m cosmic_watch.pipeline              # run forever, each source on its own interval
    python -m cosmic_watch.pipeline --once       # one pass over all sources
    python -m cosmic_watch.pipeline --once --only sdo,alerts
"""
import argparse
import logging
import time
import traceback

from . import anomaly, db
from .config import INTERVALS, RUBIN_STAMPS_PER_CLASS, STAMP_DIR
from .detectors import galaxies, stamps, sun
from .fetchers import alerts, mast, sdo

log = logging.getLogger("cosmic_watch")

# Stamp priority: the interesting classes first, junk last.
STAMP_PRIORITY = {"TDE": 0, "AGN": 1, "QSO": 1, "Blazar": 1, "SN": 2, "VS": 3, "asteroid": 4, "satellite": 5, "bogus": 6}


def _annotated_path(path):
    return path.with_name(path.stem + "_ai.jpg")


def run_sdo(con) -> int:
    n = 0
    for rec in sdo.fetch():
        if db.image_exists(con, rec["source"], rec["external_id"]):
            continue
        if rec["mode"] == "sunspots":
            img, regs, stats = sun.sunspots(rec["path"])
        else:
            n_sigma = 5.0 if rec["filter"] in ("0304", "0131", "0094") else 3.0
            img, regs, stats = sun.active_regions(rec["path"], n_sigma=n_sigma)
        out = sun.annotate(img, regs, _annotated_path(rec["path"]))
        db.insert_image(con, rec, out, stats, regs)
        log.info("SDO %s: %s", rec["filter"], stats)
        n += 1
    return n


def run_mast(con) -> int:
    n = 0
    for rec in mast.fetch(lambda src, oid: db.image_exists(con, src, oid)):
        rgb, dets, stats = galaxies.detect(rec["path"])
        out = galaxies.annotate(rgb, dets, _annotated_path(rec["path"]))
        db.insert_image(con, rec, out, stats, dets)
        log.info("%s %s: %d sources, %d galaxies", rec["source"].upper(), rec["target"],
                 stats["n_sources"], stats.get("n_galaxies", 0))
        n += 1
    return n


def _attach_stamp(con, a: dict) -> None:
    """Download cutouts, render them and score the difference image (in place on `a`)."""
    if db.alert_has_stamp(con, a["survey"], a["oid"]):
        return
    try:
        st = alerts.stamps(a["survey"], a["oid"])
    except Exception as e:                     # one bad object must not stop the run
        log.warning("stamps failed for %s %s: %s", a["survey"], a["oid"], e)
        return
    if st is None:
        return
    out_dir = STAMP_DIR / a["survey"]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{a['oid']}.png"
    stamps.render(st, path, f"{a['survey'].upper()} {a['oid']}  {a['cls']} p={a['probability']:.2f}")
    a["stamp_path"] = db.rel(path)
    a["diff_snr"] = round(stamps.diff_snr(st["difference"]), 1)


def run_alerts(con) -> int:
    n_new = 0
    errors = []

    # Rubin: newest objects; stamps for the newest few of every class.
    try:
        rubin = alerts.rubin_latest()
        rubin.sort(key=lambda a: (STAMP_PRIORITY.get(a["cls"], 9), -a["lastmjd"]))
        per_class = {}
        for a in rubin:
            quota = RUBIN_STAMPS_PER_CLASS.get(a["cls"], 2)
            if per_class.get(a["cls"], 0) < quota:
                _attach_stamp(con, a)
                per_class[a["cls"]] = per_class.get(a["cls"], 0) + 1
            n_new += db.upsert_alert(con, a)
        log.info("Rubin: %d objects", len(rubin))
    except Exception as e:
        errors.append(f"rubin: {e}")
        log.exception("Rubin fetch failed")

    # ZTF: black-hole candidates + supernovae; stamps for all of them (small list).
    try:
        ztf = alerts.ztf_candidates()
        for a in ztf:
            _attach_stamp(con, a)
            n_new += db.upsert_alert(con, a)
        log.info("ZTF: %d candidates", len(ztf))
    except Exception as e:
        errors.append(f"ztf: {e}")
        log.exception("ZTF fetch failed")

    if errors:
        raise RuntimeError("; ".join(errors) + f" (stored {n_new} new)")
    return n_new


SOURCES = {"sdo": run_sdo, "mast": run_mast, "alerts": run_alerts}


RETRY_AFTER_ERROR_S = 120


def run_source(con, name: str) -> bool:
    run_id = db.start_run(con, name)
    try:
        n = SOURCES[name](con)
        db.finish_run(con, run_id, "ok", n)
        log.info("[%s] ok, %d new", name, n)
        try:
            k = anomaly.run_for(con, name)
            if k:
                log.info("[%s] %d new anomalies", name, k)
        except Exception as e:      # anomaly checks must never break fetching
            log.warning("[%s] anomaly checks failed: %s", name, e)
        return True
    except Exception as e:
        db.finish_run(con, run_id, "error", 0, f"{e}\n{traceback.format_exc(limit=2)}")
        log.error("[%s] failed: %s", name, e)
        return False


def next_delay(name: str, ok: bool) -> float:
    """Normal interval after success; retry soon after a failure (network blips are common)."""
    return INTERVALS[name] if ok else min(RETRY_AFTER_ERROR_S, INTERVALS[name])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--once", action="store_true", help="run each source once and exit")
    ap.add_argument("--only", default=",".join(SOURCES), help="comma-separated: sdo,mast,alerts")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    names = [s.strip() for s in args.only.split(",") if s.strip()]
    unknown = set(names) - set(SOURCES)
    if unknown:
        ap.error(f"unknown source(s): {', '.join(sorted(unknown))}")

    con = db.connect()
    if args.once:
        for name in names:
            run_source(con, name)
        return

    next_due = {name: 0.0 for name in names}
    log.info("Live mode: %s", ", ".join(f"{n} every {INTERVALS[n] // 60} min" for n in names))
    while True:
        now = time.time()
        for name in names:
            if now >= next_due[name]:
                ok = run_source(con, name)
                next_due[name] = time.time() + next_delay(name, ok)
        time.sleep(15)


if __name__ == "__main__":
    main()
