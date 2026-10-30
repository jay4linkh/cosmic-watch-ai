"""Anomaly detection on everything the pipeline sees.

Sun (time-domain):
  * frame differencing between consecutive images of a channel -> sudden brightening
    in one spot (a possible flare), located and measured in sigma above the disk noise
  * robust z-scores of each channel's activity metrics against its own recent history

Deep space (population outliers):
  * Isolation Forest over every detection's morphology (brightness, size, ellipticity,
    size relative to the PSF): the sources that look least like the rest of the field

Alerts:
  * strong black-hole candidates (high-probability TDEs)
  * classifier vs. CV disagreement: "bogus" by the classifier but a strong, centred
    signal in our difference-image score

Each finding goes into the `anomalies` table with a score, a plain-English reason
and a pointer back to the object, so the agent can suggest it and the app can open it.
"""
import json
import logging

import cv2
import numpy as np
from sklearn.ensemble import IsolationForest

from . import db
from .config import ROOT

log = logging.getLogger("cosmic_watch")

METRIC_Z = 3.0              # robust z-score for a metric jump
HISTORY = 24                # samples of history per channel (~6 h at 15 min)
DEEP_PER_IMAGE = 4          # outliers kept per deep image


def _path(p):
    from pathlib import Path
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def _robust_z(x, hist):
    hist = np.asarray(hist, float)
    med = np.median(hist)
    mad = 1.4826 * np.median(np.abs(hist - med))
    scale = max(mad, 0.05 * abs(med), 1e-6)
    return (x - med) / scale


def record(con, a: dict) -> bool:
    """Insert an anomaly once (dedupe on key). Returns True if new."""
    cur = con.execute(
        """INSERT OR IGNORE INTO anomalies (key, source, kind, score, title, reason, hypotheses, ref, created_at)
           VALUES (:key, :source, :kind, :score, :title, :reason, :hypotheses, :ref, :created_at)""",
        {**a, "hypotheses": json.dumps(a.get("hypotheses", [])), "ref": json.dumps(a.get("ref", {})),
         "created_at": db.utcnow()},
    )
    con.commit()
    return cur.rowcount > 0


# ---------- Sun ----------

def _disk(gray):
    _, th = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    cs, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    (cx, cy), r = cv2.minEnclosingCircle(max(cs, key=cv2.contourArea)) if cs else ((512, 512), 400)
    m = np.zeros_like(gray, np.uint8)
    cv2.circle(m, (int(cx), int(cy)), int(r * 0.97), 1, -1)
    return m.astype(bool), (cx, cy), r


FLARE_CHANNELS = ("0131", "0094", "0171", "0193", "0304")   # EUV; flares barely show in visible light
FLARE_RISE = {"0304": 80}       # gray levels; the chromosphere flickers more
FLARE_RISE_DEFAULT = 60
FLARE_RATIO = 1.5
FLARE_AREA = 40                 # px of clearly brighter disk


def sun_flare_check(con, channel: str):
    """Compare the two newest images of a channel; flag a localized sudden brightening.

    Measured on live data, normal 10-20 min changes stay under ~50 gray levels; the Sun's
    rotation shifts sharp features, so we need a big absolute rise, a big ratio AND an area.
    """
    if channel not in FLARE_CHANNELS:
        return None
    rows = con.execute("""SELECT id, path, obs_time FROM images WHERE source='sdo' AND filter=?
                          ORDER BY obs_time DESC LIMIT 2""", (channel,)).fetchall()
    if len(rows) < 2:
        return None
    new, old = rows
    a = cv2.imread(str(_path(new["path"])), cv2.IMREAD_GRAYSCALE)
    b = cv2.imread(str(_path(old["path"])), cv2.IMREAD_GRAYSCALE)
    if a is None or b is None or a.shape != b.shape:
        return None
    disk, (cx, cy), r = _disk(a)
    a32, b32 = a.astype(np.float32), b.astype(np.float32)
    b32 *= np.median(a32[disk]) / max(np.median(b32[disk]), 1)      # remove global exposure change
    A, B = cv2.GaussianBlur(a32, (9, 9), 0), cv2.GaussianBlur(b32, (9, 9), 0)
    rise, ratio = A - B, A / (B + 8)
    rise[~disk] = 0
    ratio[~disk] = 1
    need = FLARE_RISE.get(channel, FLARE_RISE_DEFAULT)
    hot = ((rise > need * 0.66) & (ratio > 1.3)).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(hot)
    best = None
    for i in range(1, n):
        area = int(st[i, cv2.CC_STAT_AREA])
        m = lab == i
        peak_rise, peak_ratio = float(rise[m].max()), float(ratio[m].max())
        if area >= FLARE_AREA and peak_rise >= need and peak_ratio >= FLARE_RATIO:
            if best is None or peak_rise > best[1]:
                best = (i, peak_rise, peak_ratio, area, cen[i])
    if not best:
        return None
    _, peak_rise, peak_ratio, area, (x, y) = best
    lat_dir = "north" if y < cy else "south"
    lon_dir = "east" if x < cx else "west"
    return {
        "key": f"sunflare:{channel}:{new['id']}",
        "source": "sdo", "kind": "sun_brightening",
        "score": round(min(1.0, (peak_rise / need) * (peak_ratio / 3)), 3),
        "title": f"Sudden brightening on the Sun ({channel} Å)",
        "reason": (f"Between {old['obs_time'][11:16]} and {new['obs_time'][11:16]} UTC a patch in the {lat_dir}-{lon_dir} "
                   f"got {peak_ratio:.1f}× brighter (+{peak_rise:.0f} levels over {area} px), far beyond the normal flicker."),
        "hypotheses": ["a solar flare starting", "an erupting loop or filament", "an image artifact (check the next frame)"],
        "ref": {"view": "sun", "channel": channel, "image_id": new["id"], "x": int(x), "y": int(y),
                "ratio": round(peak_ratio, 2)},
    }


def sun_metric_check(con, channel: str):
    """Has this channel's activity jumped compared with its own recent history?"""
    rows = con.execute("""SELECT id, stats, obs_time FROM images WHERE source='sdo' AND filter=?
                          ORDER BY obs_time DESC LIMIT ?""", (channel, HISTORY + 1)).fetchall()
    if len(rows) < 6:                       # need some history first
        return None
    stats = [json.loads(r["stats"] or "{}") for r in rows]
    metric = "n_sunspots" if "n_sunspots" in stats[0] else "activity_index"
    series = [s.get(metric, 0) for s in stats]
    z = _robust_z(series[0], series[1:])
    if abs(z) < METRIC_Z:
        return None
    word = "jumped" if z > 0 else "dropped"
    return {
        "key": f"sunmetric:{channel}:{rows[0]['id']}",
        "source": "sdo", "kind": "sun_activity_change",
        "score": round(min(1.0, abs(z) / 8), 3),
        "title": f"Solar activity {word} ({channel})",
        "reason": f"{metric.replace('_', ' ')} {word} to {series[0]} vs a typical {np.median(series[1:]):.2f} over the last "
                  f"{len(series) - 1} images ({z:+.1f}σ).",
        "hypotheses": ["a new active region rotating into view", "flare activity", "a region decaying"],
        "ref": {"view": "sun", "channel": channel, "image_id": rows[0]["id"]},
    }


def run_sun(con) -> int:
    n = 0
    for (ch,) in con.execute("SELECT DISTINCT filter FROM images WHERE source='sdo'").fetchall():
        for check in (sun_flare_check, sun_metric_check):
            try:
                a = check(con, ch)
                if a and record(con, a):
                    n += 1
                    log.info("ANOMALY %s", a["title"])
            except Exception as e:
                log.warning("sun anomaly check %s/%s failed: %s", ch, check.__name__, e)
    return n


# ---------- deep space ----------

FEATURES = ("log brightness", "log size", "ellipticity", "size vs PSF")


def deep_outliers(con, image_id: int):
    img = con.execute("SELECT * FROM images WHERE id=?", (image_id,)).fetchone()
    dets = con.execute("SELECT id, label, x, y, a, b, theta, score FROM detections WHERE image_id=?", (image_id,)).fetchall()
    if not img or len(dets) < 40:           # too few sources to say what "normal" is
        return []
    stats = json.loads(img["stats"] or "{}")
    psf = max(stats.get("psf_r50_px", 1.5), 0.5)
    w, h = stats.get("size", [1, 1])
    X = np.array([[np.log10(max(d["score"], 1)), np.log10(max(d["a"], 0.3)), 1 - d["b"] / max(d["a"], 1e-6),
                   d["a"] / psf] for d in dets])
    forest = IsolationForest(n_estimators=200, contamination="auto", random_state=0).fit(X)
    s = -forest.score_samples(X)                       # higher = more anomalous
    order = np.argsort(s)[::-1]
    mu, sd = X.mean(0), X.std(0) + 1e-6
    out = []
    # Rank by isolation score but skip saturated stars and tiny noise blips.
    for i in order:
        d = dets[i]
        if d["label"] == "star" and X[i, 3] < 3:
            continue
        zf = (X[i] - mu) / sd
        top = np.argsort(-np.abs(zf))[:2]
        why = " and ".join(f"{'unusually high' if zf[k] > 0 else 'unusually low'} {FEATURES[k]} ({zf[k]:+.1f}σ)" for k in top)
        hyp = []
        if X[i, 2] > 0.65 and X[i, 3] > 4:
            hyp += ["a gravitational lensing arc", "an edge-on disk galaxy", "a tidal tail from a merger"]
        elif X[i, 3] > 6:
            hyp += ["a large nearby galaxy", "two overlapping galaxies (a merger)", "a bright star halo"]
        else:
            hyp += ["a rare galaxy shape", "a merger", "an artifact (cosmic ray, satellite trail)"]
        out.append({
            "key": f"deep:{image_id}:{d['id']}",
            "source": img["source"], "kind": "deep_outlier",
            "score": round(float(min(1.0, (s[i] - s.mean()) / (s.std() * 4 + 1e-6))), 3),
            "title": f"Odd one out in {img['target']}",
            "reason": f"Of {len(dets)} sources the AI found in this {img['source'].upper()} image, this {d['label']} stands out: {why}.",
            "hypotheses": hyp,
            "ref": {"view": "deep", "image_id": image_id, "u": round(d["x"] / w, 5), "v": round(d["y"] / h, 5),
                    "det_id": d["id"]},
        })
        if len(out) >= DEEP_PER_IMAGE:
            break
    return out


def run_deep(con, image_ids=None) -> int:
    ids = image_ids or [r[0] for r in con.execute("SELECT id FROM images WHERE source IN ('jwst','hst')").fetchall()]
    n = 0
    for iid in ids:
        try:
            for a in deep_outliers(con, iid):
                n += record(con, a)
        except Exception as e:
            log.warning("deep anomaly check %s failed: %s", iid, e)
    return n


# ---------- alerts ----------

def run_alerts(con) -> int:
    n = 0
    for r in con.execute("""SELECT * FROM alerts WHERE cls='TDE' AND probability >= 0.5 AND stamp_path IS NOT NULL""").fetchall():
        n += record(con, {
            "key": f"tde:{r['survey']}:{r['oid']}", "source": r["survey"], "kind": "tde_candidate",
            "score": round(float(r["probability"]), 3),
            "title": f"Strong TDE candidate {r['oid']}",
            "reason": f"The light-curve classifier gives {r['probability']:.0%} odds that a black hole is tearing a star apart here "
                      f"({r['n_det']} detections).",
            "hypotheses": ["a tidal disruption event", "a supernova in the galaxy core", "an AGN flare"],
            "ref": {"view": "bh", "survey": r["survey"], "oid": r["oid"]},
        })
    for r in con.execute("""SELECT * FROM alerts WHERE cls='bogus' AND diff_snr >= 25 AND stamp_path IS NOT NULL""").fetchall():
        n += record(con, {
            "key": f"disagree:{r['survey']}:{r['oid']}", "source": r["survey"], "kind": "classifier_disagreement",
            "score": round(float(min(1.0, r["diff_snr"] / 80)), 3),
            "title": "The classifier says bogus, the pixels disagree",
            "reason": f"Labelled 'bogus' (p={r['probability']:.2f}), yet our difference-image check finds a strong, centred new "
                      f"source (S/N {r['diff_snr']:.0f}).",
            "hypotheses": ["a real transient the classifier missed", "a fast-moving asteroid", "a bad subtraction"],
            "ref": {"view": "bh", "survey": r["survey"], "oid": r["oid"]},
        })
    return n


def run_for(con, source: str) -> int:
    """Called by the pipeline after each source run."""
    if source == "sdo":
        return run_sun(con)
    if source == "mast":
        return run_deep(con)
    if source == "alerts":
        return run_alerts(con)
    return 0
