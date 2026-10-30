"""JSON API + web app for the 3D observatory.

    uvicorn cosmic_watch.server:app --port 8000
    RUN_PIPELINE=1 uvicorn cosmic_watch.server:app   # also fetch live data in the background (deploys)
"""
import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from . import agent, anomaly, db, deepzoom
from .config import DATA, INTERVALS, ROOT
from .ml import infer as cnn
from .ml.classes import CLASSES as CNN_CLASSES, COARSE as CNN_COARSE

Image.MAX_IMAGE_PIXELS = None
log = logging.getLogger("cosmic_watch")

WEB_DIR = ROOT / "web"
THUMB_DIR = DATA / "thumbs"
THUMB_VERSION = 2   # bump when thumbnail processing changes, so browsers refetch
SOURCES = {"alerts": "Rubin + ZTF alerts", "sdo": "SDO · Sun", "mast": "JWST + Hubble"}
BH_CLASSES = ("TDE", "AGN", "QSO", "Blazar")


def _pipeline_loop():
    from . import pipeline
    con = db.connect()
    next_due = {name: 0.0 for name in pipeline.SOURCES}
    while True:
        for name in pipeline.SOURCES:
            if time.time() >= next_due[name]:
                ok = pipeline.run_source(con, name)
                next_due[name] = time.time() + pipeline.next_delay(name, ok)
        time.sleep(15)


@asynccontextmanager
async def lifespan(_app):
    db.connect().close()      # create tables on first boot
    if os.environ.get("RUN_PIPELINE") == "1":
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        threading.Thread(target=_pipeline_loop, daemon=True, name="pipeline").start()
        log.info("Background pipeline started")
    yield


app = FastAPI(title="Cosmic Watch AI", lifespan=lifespan)
app.include_router(deepzoom.router)


@app.middleware("http")
async def revalidate_app_files(request, call_next):
    """The web app's own HTML/JS/CSS: always revalidate (ETag), so an update is never stuck in a browser cache."""
    resp = await call_next(request)
    if not request.url.path.startswith(("/api/", "/media/", "/thumb/")) and "cache-control" not in resp.headers:
        resp.headers["Cache-Control"] = "no-cache"
    return resp


# ---------- helpers ----------

def rows(sql, params=()):
    con = db.connect()
    try:
        return [dict(r) for r in con.execute(sql, params).fetchall()]
    finally:
        con.close()


def media_url(p):
    """Public URL for a stored file path (relative to the project root, or absolute inside DATA)."""
    if not p:
        return None
    path = Path(p)
    path = path if path.is_absolute() else ROOT / path
    try:
        return "/media/" + path.resolve().relative_to(DATA.resolve()).as_posix()
    except ValueError:
        return None


def thumb_url(p, size=2048):
    u = media_url(p)
    return u.replace("/media/", f"/thumb/{size}/", 1) + f"?v={THUMB_VERSION}" if u else None


def mjd_iso(mjd):
    if mjd is None:
        return None
    return datetime.fromtimestamp((mjd - 40587.0) * 86400.0, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def iso(ts):
    return ts.replace(" ", "T") + "Z" if ts else None


def broker_url(survey, oid):
    return f"https://alerce.online/object/{oid}" if survey == "ztf" else f"https://lsst.fink-portal.org/{oid}"


def image_out(r):
    stats = json.loads(r.get("stats") or "{}")
    return {
        "id": r["id"], "source": r["source"], "title": r["title"], "target": r["target"],
        "instrument": r["instrument"], "filter": r["filter"], "obs_time": iso(r["obs_time"]),
        "fetched_at": iso(r["fetched_at"]), "external_id": r["external_id"], "stats": stats,
        "image_url": thumb_url(r["path"]), "annotated_url": thumb_url(r["annotated_path"]),
        "full_url": media_url(r["path"]),
    }


def alert_out(r):
    return {
        "id": r["id"], "survey": r["survey"], "oid": r["oid"], "cls": r["cls"], "classifier": r["classifier"],
        "probability": r["probability"], "last_seen": mjd_iso(r["lastmjd"]), "ra": r["ra"], "dec": r["dec"],
        "n_det": r["n_det"], "bh": bool(r["bh_candidate"]), "diff_snr": r["diff_snr"],
        "stamp_url": media_url(r["stamp_path"]), "broker_url": broker_url(r["survey"], r["oid"]),
        "first_seen": iso(r.get("first_seen")),
    }


# ---------- API ----------

@app.get("/api/summary")
def summary():
    last = {r["source"]: r for r in rows(
        "SELECT r.* FROM runs r JOIN (SELECT source, MAX(id) id FROM runs GROUP BY source) m ON r.id = m.id")}
    ok = {r["source"]: r["t"] for r in rows(
        "SELECT source, MAX(finished_at) t FROM runs WHERE status='ok' GROUP BY source")}
    sources = []
    for key, name in SOURCES.items():
        r = last.get(key)
        state = "never" if r is None else ("running" if r["finished_at"] is None else r["status"])
        sources.append({"key": key, "name": name, "state": state, "last_ok": iso(ok.get(key)),
                        "interval_min": INTERVALS[key] // 60,
                        "message": (r or {}).get("message") if state == "error" else None})

    imgs = rows("SELECT source, stats FROM images")
    deep = [json.loads(i["stats"] or "{}") for i in imgs if i["source"] in ("jwst", "hst")]
    counts = {r["k"]: r["n"] for r in rows(
        "SELECT survey || ':' || cls AS k, COUNT(*) n FROM alerts GROUP BY survey, cls")}
    newest = rows("SELECT MAX(lastmjd) m FROM alerts WHERE survey='lsst'")[0]["m"]
    sun = rows("SELECT stats FROM images WHERE source='sdo' AND filter='0171' ORDER BY obs_time DESC LIMIT 1")
    sun_stats = json.loads(sun[0]["stats"]) if sun else {}
    return {
        "server_time": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sources": sources,
        "metrics": {
            "deep_images": len(deep),
            "jwst_images": sum(1 for i in imgs if i["source"] == "jwst"),
            "hst_images": sum(1 for i in imgs if i["source"] == "hst"),
            "galaxies": sum(s.get("n_galaxies", 0) for s in deep),
            "stars": sum(s.get("n_stars", 0) for s in deep),
            "bh_candidates": sum(v for k, v in counts.items() if k.split(":")[1] in BH_CLASSES),
            "tde": sum(v for k, v in counts.items() if k.endswith(":TDE")),
            "rubin_objects": sum(v for k, v in counts.items() if k.startswith("lsst:")),
            "ztf_objects": sum(v for k, v in counts.items() if k.startswith("ztf:")),
            "active_regions": sun_stats.get("n_active_regions", 0),
            "newest_rubin": mjd_iso(newest),
        },
        "alert_classes": counts,
    }


@app.get("/api/events")
def events(limit: int = 60):
    """A merged, newest-first feed of what the pipeline found."""
    out = []
    for r in rows("SELECT id, source, target, instrument, stats, fetched_at FROM images ORDER BY id DESC LIMIT ?", (limit,)):
        s = json.loads(r["stats"] or "{}")
        if r["source"] == "sdo":
            detail = (f"{s['n_sunspots']} sunspots" if "n_sunspots" in s
                      else f"{s.get('n_active_regions', 0)} active regions")
            title = f"New Sun image · {r['instrument']}"
        else:
            detail = f"{s.get('n_galaxies', 0)} galaxies · {s.get('n_stars', 0)} stars"
            title = f"{r['source'].upper()} release · {r['target']}"
        out.append({"key": f"img{r['id']}", "ts": iso(r["fetched_at"]), "source": r["source"],
                    "kind": "image", "title": title, "detail": detail, "ref": r["id"]})
    for r in rows("""SELECT id, survey, oid, cls, probability, first_seen FROM alerts
                     WHERE bh_candidate=1 OR cls IN ('SN','TDE') ORDER BY first_seen DESC, id DESC LIMIT ?""", (limit,)):
        bh = r["cls"] in BH_CLASSES
        out.append({"key": f"al{r['id']}", "ts": iso(r["first_seen"]), "source": r["survey"],
                    "kind": "blackhole" if bh else "transient",
                    "title": f"{'Black-hole activity' if bh else 'Supernova'} · {r['cls']}",
                    "detail": f"{r['survey'].upper()} {r['oid']} · p={r['probability']:.2f}", "ref": r["id"]})
    for r in rows("SELECT id, source, status, n_new, finished_at, message FROM runs WHERE finished_at IS NOT NULL ORDER BY id DESC LIMIT 20"):
        ok = r["status"] == "ok"
        out.append({"key": f"run{r['id']}", "ts": iso(r["finished_at"]), "source": r["source"], "kind": "run" if ok else "error",
                    "title": f"{SOURCES.get(r['source'], r['source'])} · {'fetched' if ok else 'failed'}",
                    "detail": f"{r['n_new']} new" if ok else (r["message"] or "").splitlines()[0][:90], "ref": None})
    out.sort(key=lambda e: e["ts"] or "", reverse=True)
    return out[:limit]


@app.get("/api/images")
def images(source: str = "jwst,hst", limit: int = 30):
    srcs = [s for s in source.split(",") if s]
    marks = ",".join("?" * len(srcs))
    return [image_out(r) for r in rows(
        f"SELECT * FROM images WHERE source IN ({marks}) ORDER BY obs_time DESC, id DESC LIMIT ?", (*srcs, limit))]


@app.get("/api/images/{image_id}/detections")
def detections(image_id: int, limit: int = Query(4000, le=20000)):
    img = rows("SELECT * FROM images WHERE id=?", (image_id,))
    if not img:
        raise HTTPException(404)
    stats = json.loads(img[0]["stats"] or "{}")
    w, h = stats.get("size") or [1024, 1024]
    dets = rows("""SELECT id, label, x, y, a, b, theta, score, cnn_label, cnn_conf FROM detections
                   WHERE image_id=? ORDER BY score DESC LIMIT ?""", (image_id, limit))
    labels = sorted({d["label"] for d in dets})
    idx = {l: i for i, l in enumerate(labels)}
    return {
        "image": image_out(img[0]), "width": w, "height": h, "labels": labels,
        "cnn_classes": CNN_CLASSES,
        # compact rows: [u, v, a/w, b/w, theta, label_index, score, cnn_class (-1 = none), cnn_conf, detection_id]
        "rows": [[round(d["x"] / w, 5), round(d["y"] / h, 5), round(d["a"] / w, 5), round(d["b"] / w, 5),
                  round(d["theta"] or 0, 3), idx[d["label"]], round(d["score"] or 0, 2),
                  CNN_CLASSES.index(d["cnn_label"]) if d["cnn_label"] in CNN_CLASSES else -1,
                  round(d["cnn_conf"] or 0, 3), d["id"]] for d in dets],
    }


@app.get("/api/detections/{det_id}/cnn")
def detection_cnn(det_id: int):
    """Full CNN probabilities for one detection (Inspect mode)."""
    r = rows("SELECT cnn_label, cnn_conf, cnn_probs FROM detections WHERE id=?", (det_id,))
    if not r or not r[0]["cnn_probs"]:
        return {"available": False}
    probs = json.loads(r[0]["cnn_probs"])
    order = sorted(range(len(probs)), key=lambda i: -probs[i])
    return {"available": True, "label": r[0]["cnn_label"], "conf": r[0]["cnn_conf"],
            "top": [[CNN_CLASSES[i], probs[i]] for i in order[:4]],
            "coarse": {k: round(sum(probs[i] for i in idx), 4) for k, idx in CNN_COARSE.items()},
            "model": cnn.version()}


@app.get("/api/images/{image_id}/crop")
def crop(image_id: int, u: float = Query(ge=0, le=1), v: float = Query(ge=0, le=1),
         r: float = Query(0.02, gt=0, le=0.5), size: int = Query(512, ge=128, le=1024)):
    """Full-resolution close-up around (u, v): the real pixels behind one detection."""
    img = rows("SELECT path FROM images WHERE id=?", (image_id,))
    if not img:
        raise HTTPException(404)
    src = Path(img[0]["path"])
    src = src if src.is_absolute() else ROOT / src
    if not src.is_file():
        raise HTTPException(404)
    out = THUMB_DIR / "crops" / f"{image_id}_{u:.4f}_{v:.4f}_{r:.4f}_{size}.jpg"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        im = Image.open(src)
        W, H = im.size
        half = max(int(r * W), 40)
        cx, cy = int(u * W), int(v * H)
        box = (max(0, cx - half), max(0, cy - half), min(W, cx + half), min(H, cy + half))
        im.crop(box).convert("RGB").resize((size, size), Image.LANCZOS).save(out, quality=90)
    return FileResponse(out, headers={"Cache-Control": "public, max-age=86400"})


@app.get("/api/sun")
def sun():
    latest = rows("""SELECT i.* FROM images i JOIN (SELECT filter, MAX(id) id FROM images WHERE source='sdo' GROUP BY filter) m
                     ON i.id = m.id ORDER BY i.filter""")
    out = []
    for r in latest:
        o = image_out(r)
        o["image_url"] = media_url(r["path"])        # SDO images are only 1024 px; send the original
        o["annotated_url"] = media_url(r["annotated_path"])
        o["regions"] = rows("SELECT label, x, y, a, b, score FROM detections WHERE image_id=? ORDER BY (a*b) DESC LIMIT 40",
                            (r["id"],))
        out.append(o)
    return out


@app.get("/api/alerts")
def alerts(bh: bool = False, survey: str | None = None, cls: str | None = None,
           with_stamp: bool = True, limit: int = Query(60, le=500)):
    where, params = ["1=1"], []
    if bh:
        where.append("bh_candidate=1")
    if survey:
        where.append("survey=?"); params.append(survey)
    if cls:
        where.append("cls=?"); params.append(cls)
    if with_stamp:
        where.append("stamp_path IS NOT NULL")
    order = "CASE cls WHEN 'TDE' THEN 0 ELSE 1 END, lastmjd DESC" if bh else "lastmjd DESC"
    return [alert_out(r) for r in rows(
        f"SELECT * FROM alerts WHERE {' AND '.join(where)} ORDER BY {order} LIMIT ?", (*params, limit))]


class LabelIn(BaseModel):
    key: str = Field(max_length=120)
    kind: str = Field(pattern="^(galaxy|bh)$")
    label: str = Field(max_length=40)
    ai_label: str | None = Field(default=None, max_length=40)


@app.post("/api/labels")
def add_label(body: LabelIn):
    """Save a person's classification (latest answer wins). These become training labels."""
    con = db.connect()
    try:
        con.execute("""INSERT INTO user_labels (key, kind, label, ai_label, created_at) VALUES (?,?,?,?,?)
                       ON CONFLICT (key) DO UPDATE SET label=excluded.label, created_at=excluded.created_at""",
                    (body.key, body.kind, body.label, body.ai_label, db.utcnow()))
        con.commit()
        total = con.execute("SELECT COUNT(*) FROM user_labels").fetchone()[0]
    finally:
        con.close()
    return {"ok": True, "total": total}


@app.get("/api/labels")
def label_stats():
    by = rows("SELECT kind, label, COUNT(*) n FROM user_labels GROUP BY kind, label ORDER BY n DESC")
    return {"total": sum(r["n"] for r in by), "by_label": by,
            "recent": rows("SELECT key, kind, label, ai_label, created_at FROM user_labels ORDER BY id DESC LIMIT 20")}


class FeedbackIn(BaseModel):
    id: int
    action: str = Field(pattern="^(look|dismiss|label)$")


@app.get("/api/agent/suggestion")
def agent_suggestion(exclude: str = ""):
    con = db.connect()
    try:
        ex = {int(x) for x in exclude.split(",") if x.strip().isdigit()}
        return {"suggestion": agent.suggest(con, ex), "stats": agent.stats(con)}
    finally:
        con.close()


@app.post("/api/agent/feedback")
def agent_feedback(body: FeedbackIn):
    con = db.connect()
    try:
        return agent.feedback(con, body.id, body.action)
    finally:
        con.close()


@app.get("/api/anomalies")
def anomalies_list(limit: int = Query(50, le=500)):
    out = rows("SELECT * FROM anomalies ORDER BY id DESC LIMIT ?", (limit,))
    for r in out:
        r["hypotheses"] = json.loads(r["hypotheses"] or "[]")
        r["ref"] = json.loads(r["ref"] or "{}")
    return out


# ---------- files ----------

def _safe(rel: str) -> Path:
    p = (DATA / rel).resolve()
    if not p.is_relative_to(DATA.resolve()) or not p.is_file():
        raise HTTPException(404)
    return p


@app.get("/media/{rel:path}")
def media(rel: str):
    return FileResponse(_safe(rel), headers={"Cache-Control": "public, max-age=86400"})


def _blacken_footprint(im: Image.Image) -> Image.Image:
    """MAST previews pad outside the footprint with flat white (JWST); make it black so it doesn't glow in 3D."""
    import numpy as np
    from .detectors.galaxies import _footprint_mask
    rgb = np.asarray(im).copy()
    gray = np.asarray(im.convert("L"), dtype=np.float32)
    outside = _footprint_mask(gray)
    rgb[outside] = 0
    return Image.fromarray(rgb)


@app.get("/thumb/{size}/{rel:path}")
def thumb(size: int, rel: str):
    """Downscaled copy for WebGL textures (MAST previews can be 5000 px)."""
    size = max(256, min(size, 4096))
    src = _safe(rel)
    out = THUMB_DIR / str(size) / (rel.rsplit(".", 1)[0] + ".jpg")
    if not out.exists() or out.stat().st_mtime < src.stat().st_mtime:
        out.parent.mkdir(parents=True, exist_ok=True)
        im = Image.open(src)
        im.thumbnail((size, size), Image.LANCZOS)
        im = im.convert("RGB")
        if rel.startswith(("images/jwst/", "images/hst/")) and "_ai" not in rel:
            im = _blacken_footprint(im)
        im.save(out, quality=88)
    return FileResponse(out, headers={"Cache-Control": "public, max-age=86400"})


@app.get("/healthz")
def healthz():
    return {"ok": True}


app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
