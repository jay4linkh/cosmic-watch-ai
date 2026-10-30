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


# Checked on live data: confident 'Merging' calls are real close pairs; 'Disturbed' also catches
# star spikes and trails (Galaxy10 has no 'not a galaxy' class), so only Merging, and only when sure.
MERGER_CONF = 0.9


def cnn_mergers(con, image_id: int):
    """Galaxies our CNN thinks are merging or disturbed: rare and scientifically interesting."""
    img = con.execute("SELECT source, target, stats FROM images WHERE id=?", (image_id,)).fetchone()
    if not img:
        return []
    w, h = json.loads(img["stats"] or "{}").get("size", [1, 1])
    out = []
    for d in con.execute("""SELECT id, x, y, cnn_label, cnn_conf FROM detections WHERE image_id=?
                            AND cnn_label = 'Merging' AND cnn_conf >= ?
                            ORDER BY cnn_conf DESC LIMIT 3""", (image_id, MERGER_CONF)).fetchall():
        out.append({
            "key": f"cnnmerger:{image_id}:{d['id']}", "source": img["source"], "kind": "cnn_merger",
            "score": round(float(d["cnn_conf"]), 3),
            "title": f"Possible galaxy collision in {img['target']}",
            "reason": f"Our CNN (trained on Galaxy Zoo) is {d['cnn_conf']:.0%} sure this galaxy is merging with another.",
            "hypotheses": ["two galaxies colliding", "a galaxy with tidal tails", "two unrelated galaxies overlapping in projection"],
            "ref": {"view": "deep", "image_id": image_id, "u": round(d["x"] / w, 5), "v": round(d["y"] / h, 5), "det_id": d["id"]},
        })
    return out


def run_deep(con, image_ids=None) -> int:
    ids = image_ids or [r[0] for r in con.execute("SELECT id FROM images WHERE source IN ('jwst','hst')").fetchall()]
    n = 0
    for iid in ids:
        try:
            for a in deep_outliers(con, iid) + cnn_mergers(con, iid):
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


# ---------- space weather, gravitational waves, bursts, asteroids, Mars ----------

FLARE_LETTERS = {"M": 0.6, "X": 0.95}


def run_spacewx(con) -> int:
    _, sw = db.get_feed(con, "spacewx")
    if not sw:
        return 0
    n = 0
    for f in sw.get("flares") or []:
        cls = f.get("class") or ""
        if cls[:1] in FLARE_LETTERS:
            n += record(con, {
                "key": f"flare:{f['begin']}", "source": "spacewx", "kind": "solar_flare",
                "score": FLARE_LETTERS[cls[0]], "title": f"{cls} solar flare",
                "reason": f"GOES saw an {cls} flare peak at {(f.get('max') or '')[11:16]} UTC: a strong burst of X-rays that can "
                          "cause radio blackouts on Earth's day side.",
                "hypotheses": ["a flare from a complex sunspot group", "the start of a coronal mass ejection",
                               "more flares from the same region"],
                "ref": {"view": "sun"}})
    kp = ((sw.get("kp") or {}).get("now") or {}).get("kp")
    if kp is not None and kp >= 5:
        g = min(5, int(kp) - 4)
        n += record(con, {
            "key": f"storm:{(sw['kp']['now']['time'] or '')[:13]}", "source": "spacewx", "kind": "geomagnetic_storm",
            "score": min(1.0, 0.5 + 0.1 * g), "title": f"Geomagnetic storm (G{g}, Kp {kp:.1f})",
            "reason": f"The planetary K index is {kp:.1f}: Earth's magnetic field is being shaken by the solar wind. Aurora may be "
                      "visible far from the poles.",
            "hypotheses": ["a coronal mass ejection arriving", "a fast solar-wind stream from a coronal hole",
                           "a southward magnetic field (Bz) in the wind"],
            "ref": {"view": "sun"}})
    for cam, l in (sw.get("lasco") or {}).items():
        c = (l or {}).get("cme") or {}
        if c.get("area_frac", 0) >= 0.02:
            n += record(con, {
                "key": f"cme:{cam}:{l['time']}", "source": "spacewx", "kind": "cme_candidate",
                "score": c["score"], "title": f"Possible CME in LASCO {cam.upper()}",
                "reason": f"Our running-difference check found a new bright front covering {c['area_frac'] * 100:.1f}% of the "
                          f"coronagraph field, at position angle {c['angle_deg']} degrees.",
                "hypotheses": ["a coronal mass ejection", "a bright streamer change", "a comet or a camera artifact"],
                "ref": {"view": "sun", "lasco": cam}})
    return n


GW_WHAT = {"BBH": "two black holes merged", "BNS": "two neutron stars merged",
           "NSBH": "a black hole swallowed a neutron star", "Terrestrial": "it might be noise on Earth"}


def run_gw(con) -> int:
    _, g = db.get_feed(con, "gw")
    n = 0
    for e in (g or {}).get("events", [])[:15]:
        cls = e.get("class")
        p = (e.get("classification") or {}).get(cls or "", 0)
        what = GW_WHAT.get(cls, "the type is not public")
        dist = f" about {e['distance_mpc'] * 3.262 / 1000:.1f} billion light-years away" if e.get("distance_mpc") else ""
        prob = f" ({p:.0%} probability)" if p else ""
        n += record(con, {
            "key": f"gw:{e['id']}", "source": "gw", "kind": "gw_event",
            "score": 0.95 if cls in ("BNS", "NSBH") else 0.7, "title": f"Gravitational waves: {e['id']}",
            "reason": f"LIGO/Virgo/KAGRA detected a ripple in spacetime on {e['time'][:10]}: {what}{dist}{prob}.",
            "hypotheses": ["a binary black-hole merger", "a neutron-star collision (look for light!)",
                           "a black hole eating a neutron star"],
            "ref": {"view": "bh", "gw": e["id"]}})
    return n


GCN_HYP = {"grb": ["a massive star collapsing into a black hole", "two neutron stars merging", "a magnetar giant flare"],
           "neutrino": ["a blazar jet", "a tidal disruption event", "a chance cosmic-ray coincidence"],
           "xray": ["a gamma-ray-burst cousin", "a stellar flare", "a newborn magnetar"],
           "frb": ["a magnetar in another galaxy", "a young neutron star", "something unexplained"]}


def run_gcn(con) -> int:
    _, g = db.get_feed(con, "gcn")
    n = 0
    for e in (g or {}).get("events", [])[:25]:
        if e["kind"] not in GCN_HYP:
            continue
        z = f" Its redshift is z = {e['redshift']}: light that left long before Earth existed." if e.get("redshift") else ""
        latest = e["circulars"][-1]["subject"][:110]
        n += record(con, {
            "key": f"gcn:{e['event']}", "source": "gcn", "kind": f"gcn_{e['kind']}",
            "score": round(min(1.0, 0.45 + 0.03 * e["n"] + (0.2 if e.get("redshift") else 0)), 3),
            "title": f"{e['kind_label']}: {e['event']}",
            "reason": f"{e['n']} report(s) from observatories worldwide; latest: '{latest}'.{z}",
            "hypotheses": GCN_HYP[e["kind"]],
            "ref": {"view": "bh", "gcn": e["event"]}})
    return n


def run_neo(con) -> int:
    _, d = db.get_feed(con, "neo")
    n = 0
    for a in (d or {}).get("approaches", []):
        if a["dist_ld"] <= 1.0:
            size = f"{a['size_m'][0]}-{a['size_m'][1]} m" if a.get("size_m") else "unknown size"
            n += record(con, {
                "key": f"neo:{a['des']}:{a['time']}", "source": "neo", "kind": "neo_close",
                "score": round(min(1.0, 1.1 - a["dist_ld"] * 0.6), 3),
                "title": f"Asteroid {a['name']} inside the Moon's orbit",
                "reason": f"It passes {a['dist_ld']:.2f} lunar distances from Earth ({a['dist_km']:,} km) on {a['time']} UTC at "
                          f"{a['v_kms']} km/s. Estimated size {size}.",
                "hypotheses": ["a harmless close flyby", "a small rock discovered days ago", "a target for radar imaging"],
                "ref": {"view": "neo", "des": a["des"]}})
    return n


MARS_HYP = {"Light-toned veins": ["mineral veins left by ancient water", "a dust streak", "a lighting effect"],
            "Layers": ["sedimentary layers from an ancient lake", "volcanic layers", "wind-blown deposits"],
            "Drill hole": ["a fresh sample drill hole", "a crater-like rock pit", "a misclassified shadow"]}


def run_mars(con) -> int:
    from .ml import mars as mars_cv
    n = 0
    for r in con.execute("SELECT id, title, stats FROM images WHERE source='mars' ORDER BY id DESC LIMIT 60").fetchall():
        s = json.loads(r["stats"] or "{}")
        if s.get("scene") in mars_cv.SCIENCE and (s.get("scene_conf") or 0) >= 0.6:
            n += record(con, {
                "key": f"mars:{r['id']}", "source": "mars", "kind": "mars_feature",
                "score": round(s["scene_conf"], 3), "title": f"{s['scene']} on Mars",
                "reason": f"Our Mars CNN thinks this {s.get('rover_name')} image (sol {s.get('sol')}) shows "
                          f"{mars_cv.NOTES.get(s['scene'], s['scene'])} ({s['scene_conf']:.0%}).",
                "hypotheses": MARS_HYP.get(s["scene"], ["a science target", "an engineering image", "a misclassification"]),
                "ref": {"view": "mars", "image_id": r["id"]}})
    return n


def run_for(con, source: str) -> int:
    """Called by the pipeline after each source run."""
    extra = {"spacewx": run_spacewx, "gw": run_gw, "gcn": run_gcn, "neo": run_neo, "mars": run_mars}
    if source in extra:
        return extra[source](con)
    if source == "sdo":
        return run_sun(con)
    if source == "mast":
        return run_deep(con)
    if source == "alerts":
        return run_alerts(con)
    return 0
