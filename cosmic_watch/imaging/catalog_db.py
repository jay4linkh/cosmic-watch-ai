"""Run the science detector on an image's FITS mosaic and fold the result into the detections table.

Nothing is thrown away: a detection the old (JPEG preview) detector already had keeps its row id, so labels, CNN
results and archive matches that point at it stay valid; it gains measured columns. New sources are inserted.
Old detections the science data does not confirm stay, marked det_mode = 'preview-only'.
"""
import json
import logging

import numpy as np

from .. import db
from ..config import DATA
from . import detect

log = logging.getLogger(__name__)
FITS_DIR = DATA / "science" / "fits"
CAT_DIR = DATA / "science" / "catalogs"


def table_path(image_id):
    return CAT_DIR / f"{image_id}_{detect.VERSION}.ecsv"


def report_path(image_id):
    return CAT_DIR / f"{image_id}_{detect.VERSION}.json"


def load_report(image_id):
    p = report_path(image_id)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _to_frame(cat, shape, size, flip):
    """Mosaic pixels -> the detections' frame (the preview grid; FITS rows run bottom-up when flip)."""
    hf, wf = shape
    w, h = size
    x = (cat["x"] + 0.5) * w / wf - 0.5
    yf = (hf - 1 - cat["y"]) if flip else cat["y"]
    y = (yf + 0.5) * h / hf - 0.5
    return x, y, cat["a"] * w / wf, cat["b"] * w / wf, (-cat["theta"] if flip else cat["theta"])


def _num(v):
    return None if v is None or not np.isfinite(v) else float(v)


def apply(con, image_id, cat, rep):
    from scipy.spatial import cKDTree
    img = con.execute("SELECT stats FROM images WHERE id=?", (image_id,)).fetchone()
    stats = json.loads(img["stats"] or "{}")
    size = stats.get("size") or [rep["shape"][1], rep["shape"][0]]
    old = con.execute("SELECT id, x, y, a, score, origin FROM detections WHERE image_id=? AND x IS NOT NULL", (image_id,)).fetchall()
    n = len(cat["x"])
    pix = rep["meta"].get("pixscale_arcsec") or float("nan")
    best = None
    for flip in (True, False):                      # the orientation is checked against the existing detections, not assumed
        x, y, a, b, th = _to_frame(cat, rep["shape"], size, flip)
        if old and n:
            d, j = cKDTree(np.c_[x, y]).query(np.array([[o["x"], o["y"]] for o in old]))
            ok = d < np.maximum(2.5, 0.6 * np.array([o["a"] or 1 for o in old]))
        else:
            j, ok = np.array([], int), np.array([], bool)
        if best is None or ok.sum() > best[0].sum():
            best = (ok, j, x, y, a, b, th, flip)
    ok, j, x, y, a, b, th, flip = best
    used, ratio = {}, []
    for o, m, k in zip(old, ok, j):                 # one new source per old row: the first claim wins
        if m and int(k) not in used:
            used[int(k)] = o
            if cat["flux"][k] > 0 and o["score"]:
                ratio.append(o["score"] / cat["flux"][k])
    scale = float(np.median(ratio)) if ratio else 1.0     # keeps `score` (brightness ordering) on the old scale
    upd, ins = [], []
    for k in range(n):
        row = (str(cat["label"][k]), _num(x[k]), _num(y[k]), _num(a[k]), _num(b[k]), _num(th[k]), _num(cat["mag"][k]), _num(cat["mag_err"][k]),
               _num(cat["snr"][k]), _num(cat["r50_px"][k] * pix), str(cat["size_class"][k]), str(cat["mode"][k]), _num(cat["ra"][k]), _num(cat["dec"][k]))
        if k in used:
            upd.append(row + (used[k]["id"],))
        else:
            ins.append((image_id,) + row + (_num(cat["flux"][k] * scale),))
    con.executemany("""UPDATE detections SET label=?, x=?, y=?, a=?, b=?, theta=?, mag=?, mag_err=?, snr=?, r50_arcsec=?, size_class=?, det_mode=?,
                       ra=?, dec=? WHERE id=?""", upd)
    con.executemany("""INSERT INTO detections (image_id, label, x, y, a, b, theta, mag, mag_err, snr, r50_arcsec, size_class, det_mode, ra, dec, score, origin)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'science')""", ins)
    keep = {o["id"] for o in used.values()}
    # not confirmed this time: preview detections become 'preview-only', sources from an earlier science run become 'stale'
    con.executemany("UPDATE detections SET det_mode=? WHERE id=?",
                    [("stale" if o["origin"] == "science" else "preview-only", o["id"]) for o in old if o["id"] not in keep])
    return summarize(con, image_id, rep, flip)


def summarize(con, image_id, rep, flip=True):
    """Counts for the image's stats, from what is in the table now (so re-running gives the same numbers)."""
    stats = json.loads(con.execute("SELECT stats FROM images WHERE id=?", (image_id,)).fetchone()["stats"] or "{}")
    live = "image_id=? AND (det_mode IS NULL OR det_mode NOT IN ('preview-only', 'stale'))"
    counts = {r["label"]: r["n"] for r in con.execute(f"SELECT label, COUNT(*) n FROM detections WHERE {live} GROUP BY label", (image_id,))}
    one = lambda q: con.execute(q, (image_id,)).fetchone()[0]  # noqa: E731
    comp = rep.get("completeness") or {}
    stats.update(n_sources=sum(counts.values()), n_galaxies=sum(v for k, v in counts.items() if "galaxy" in k), n_stars=counts.get("star", 0), counts=counts)
    stats["sci"] = {"version": detect.VERSION, "n_sources": rep["n_sources"],
                    "confirmed_old": one(f"SELECT COUNT(*) FROM detections WHERE {live} AND det_mode IS NOT NULL AND origin IS NULL"),
                    "new": one(f"SELECT COUNT(*) FROM detections WHERE {live} AND origin='science'"),
                    "unconfirmed_old": one("SELECT COUNT(*) FROM detections WHERE image_id=? AND det_mode='preview-only'"),
                    "size_classes": rep["size_classes"], "psf_fwhm_arcsec": rep.get("psf_fwhm_arcsec"), "mag_range": rep.get("mag_range"),
                    "mag_50": {c: v.get("mag_50") for c, v in (comp.get("classes") or {}).items()}, "flipped_rows": flip}
    con.execute("UPDATE images SET stats=? WHERE id=?", (json.dumps(stats), image_id))
    con.commit()
    return stats["sci"]


def run(image_id, con=None, with_completeness=True):
    """Detect on the stored FITS, write the table and report, update the database. Returns the report."""
    own = con is None
    con = con or db.connect()
    try:
        fits = FITS_DIR / f"{image_id}.fits"
        if not fits.exists():
            raise FileNotFoundError(f"no science FITS for image {image_id} (open it in deep zoom once to fetch it)")
        cat, rep = detect.detect(fits, with_completeness=with_completeness)
        CAT_DIR.mkdir(parents=True, exist_ok=True)
        detect.write_table(cat, rep, table_path(image_id))
        rep["database"] = apply(con, image_id, cat, rep)
        report_path(image_id).write_text(json.dumps(rep, default=str), encoding="utf-8")
        try:
            from ..ml import infer as cnn
            if cnn.available():
                cnn.classify_image(con, image_id)
        except Exception as e:  # noqa: BLE001  (the CNN is an extra)
            log.warning("CNN on new detections of image %s failed: %s", image_id, e)
        log.info("science catalogue for image %s: %s", image_id, rep["database"])
        return rep
    finally:
        if own:
            con.close()
