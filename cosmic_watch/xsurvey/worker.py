"""Background cross-survey batches for the live pipeline.

The pipeline runs sources one after another, and a single catalogue query can take a minute (Gaia, JPL), so each
batch runs in its own thread with its own database connection; the pipeline only starts it and moves on.
Order of work: newest alerts first (the interesting, time-critical part), then GCN events with arcsecond positions,
then JWST/Hubble images (WCS -> sky positions -> one field query -> every bright detection).
"""
import logging
import threading
import time

from .. import db
from . import engine, wcs

log = logging.getLogger("cosmic_watch")
BUDGET_S = 8 * 60
ALERTS_PER_BATCH = 25
IMAGES_PER_BATCH = 2
DETECTIONS_PER_IMAGE = 200
_lock = threading.Lock()
STATE = {"running": False, "last": None}


def _todo_alerts(con, n):
    return con.execute("""SELECT a.survey, a.oid FROM alerts a
                          LEFT JOIN astronomical_objects o ON o.ref = 'alert:' || a.survey || ':' || a.oid
                          WHERE a.ra IS NOT NULL AND o.id IS NULL
                          ORDER BY (a.cls IN ('TDE','SN','SNIa','SNII','SLSN','AGN','QSO','Blazar')) DESC, a.lastmjd DESC
                          LIMIT ?""", (n,)).fetchall()


def _todo_images(con, n):
    return con.execute("""SELECT i.id FROM images i WHERE i.source IN ('jwst','hst')
                          AND EXISTS (SELECT 1 FROM detections d WHERE d.image_id = i.id)
                          AND NOT EXISTS (SELECT 1 FROM astronomical_objects o WHERE o.ref LIKE 'detection:' || i.id || ':%')
                          AND NOT EXISTS (SELECT 1 FROM feeds f WHERE f.key = 'xsurvey_noimg:' || i.id)
                          ORDER BY i.id DESC LIMIT ?""", (n,)).fetchall()


def batch():
    """One budgeted pass. Returns counts."""
    con = db.connect()
    t0, out = time.time(), {"alerts": 0, "gcn": 0, "images": 0, "detections": 0, "errors": 0}
    try:
        for a in _todo_alerts(con, ALERTS_PER_BATCH):
            if time.time() - t0 > BUDGET_S:
                break
            try:
                t = engine.alert_target(con, a["survey"], a["oid"])
                if t:
                    engine.analyze(t, con)
                    out["alerts"] += 1
            except Exception as e:  # noqa: BLE001
                out["errors"] += 1
                log.warning("xsurvey alert %s failed: %s", a["oid"], e)
        _, g = db.get_feed(con, "gcn")
        for ev in (g or {}).get("events", [])[:20]:
            t = engine.gcn_target(ev)
            if not t or t["err"] > 30 or engine.load(con, t["ref"]) or time.time() - t0 > BUDGET_S:
                continue
            try:
                engine.analyze(t, con)
                out["gcn"] += 1
            except Exception as e:  # noqa: BLE001
                out["errors"] += 1
                log.warning("xsurvey GCN %s failed: %s", ev["event"], e)
        for r in _todo_images(con, IMAGES_PER_BATCH):
            if time.time() - t0 > BUDGET_S:
                break
            try:
                wcs.locate_image(con, r["id"])
                res = engine.analyze_image(con, r["id"], limit=DETECTIONS_PER_IMAGE)
                if not res["n"]:
                    db.put_feed(con, f"xsurvey_noimg:{r['id']}", {"reason": "no WCS / no located detections"})
                db.put_feed(con, f"xsurvey_image:{r['id']}", res)
                out["images"] += 1
                out["detections"] += res["n"]
            except Exception as e:  # noqa: BLE001
                out["errors"] += 1
                log.warning("xsurvey image %s failed: %s", r["id"], e)
    finally:
        con.close()
    out["seconds"] = round(time.time() - t0, 1)
    return out


def _run():
    try:
        res = batch()
        STATE["last"] = {**res, "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        log.info("[xsurvey] %s", res)
    except Exception as e:  # noqa: BLE001
        log.error("[xsurvey] batch failed: %s", e)
        STATE["last"] = {"error": str(e)[:200]}
    finally:
        STATE["running"] = False


def start():
    """Start a batch in the background unless one is running. Returns True if started."""
    with _lock:
        if STATE["running"]:
            return False
        STATE["running"] = True
    threading.Thread(target=_run, daemon=True, name="xsurvey").start()
    return True


def stats(con):
    rows = con.execute("SELECT kind, verdict, COUNT(*) n, AVG(verdict_conf) c FROM astronomical_objects GROUP BY kind, verdict").fetchall()
    by = {}
    for r in rows:
        by.setdefault(r["kind"], {})[r["verdict"]] = {"n": r["n"], "mean_conf": round(r["c"] or 0, 3)}
    agree = con.execute("""SELECT SUM(json_extract(report, '$.cnn_agrees') = 1), SUM(json_extract(report, '$.cnn_agrees') = 0)
                           FROM astronomical_objects WHERE kind='detection'""").fetchone()
    a, d = agree[0] or 0, agree[1] or 0
    val = None
    try:
        import json

        from .validate import OUT
        if OUT.exists():
            v = json.loads(OUT.read_text(encoding="utf-8"))
            val = {k: v[k] for k in ("precision", "recall", "n", "calibration", "created_at", "answer_key")}
    except Exception:  # noqa: BLE001
        pass
    return {"objects": by, "total": sum(r["n"] for r in rows),
            "labels_vs_archives": {"agree": a, "disagree": d, "agreement": round(a / (a + d), 3) if a + d else None},
            "validation": val, "worker": STATE, "engine": engine.ENGINE_VERSION}
