"""Active learning: the galaxy CNN gets better the longer the project runs.

The loop (driven by the pipeline's "learn" source, every 20 min):

1. ASK   Every CNN prediction gets an acquisition score: how unsure the CNN is (entropy, top-2 margin) and
         whether the source might not be a galaxy at all (nebula field, emission-line filter, flat light).
         The most informative ones form the "teach me" queue; the agent asks you about them.
2. LEARN When enough new labels have arrived (or for the first round), a short fine-tune runs in a background
         process on the GPU, from the current champion model, on:
           * Galaxy10 DECaLS (the original training set, so it does not forget)
           * your labels (fine class, coarse group, or "not a galaxy": star / gas / artifact)
           * weak labels: bright detector-confirmed stars and high-confidence nebula patches -> "Not a galaxy"
           * pseudo-labels: galaxies the CNN is already very sure of in real deep fields (keeps the new class honest)
         The model gains an 11th class, "Not a galaxy", which Galaxy10 never had.
3. GATE  Champion vs challenger on held-out data: Galaxy10 validation accuracy, held-out user labels,
         not-a-galaxy recall and galaxy false-positive rate. The challenger is promoted only if it is not worse.
4. SWAP  The promoted model is written to data/models/galaxy_cnn_live.onnx; inference reloads it automatically
         and the newest images are re-classified. Every round (promoted or not) is logged in the registry.

    python -m cosmic_watch.ml.active train      # one round now (what the pipeline launches)
    python -m cosmic_watch.ml.active queue      # print the current "teach me" queue
"""
import hashlib
import json
import logging
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .. import db
from ..config import DATA, ROOT
from .classes import CLASSES, CLASSES_ALL, COARSE, NOT_GALAXY, TEACH_LABELS
from .cutouts import cutout

log = logging.getLogger("cosmic_watch")
MODELS = DATA / "models"
LIVE = "galaxy_cnn_live"
REGISTRY = MODELS / "registry.json"
STATE = MODELS / "training.json"
TRAIN_LOG = MODELS / "training.log"
MIN_NEW_LABELS = 15            # new human labels that trigger a round
REFRESH_HOURS = 24             # otherwise refresh weak labels once a day if new images arrived
SIZE = 224
NG = 10                        # index of "Not a galaxy"
GROUPS = list(COARSE) + ["not-galaxy"]
LEGACY = {"spiral": "spiral", "elliptical": "elliptical", "edge-on": "edge-on", "merger": "merger",
          "star": "star", "artifact": "artifact", "gas": "gas"}


# ---------- small helpers ----------

def _read(p, default):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return default


def _write(p, data):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(p).with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(p)


def registry():
    return _read(REGISTRY, [])


def _holdout(key, pct=20):
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % 100 < pct


def _solar(target):
    from ..fetchers.mast import is_solar_system
    return is_solar_system(target or "")


# ---------- 1. ASK: the teach-me queue ----------

def acquisition(probs):
    p = np.asarray(probs, dtype=np.float64)
    p = p / max(p.sum(), 1e-9)
    ent = float(-(p * np.log(np.clip(p, 1e-9, 1))).sum() / math.log(len(p)))
    top = np.sort(p)[::-1]
    margin = float(top[0] - top[1])
    return 0.5 * ent + 0.5 * (1 - margin), ent, margin


def queue(con, n=24, per_image=4):
    """The detections the CNN would learn the most from a human answer on."""
    from .explain_lite import field_suspicion
    labelled = {r[0] for r in con.execute("SELECT key FROM user_labels WHERE kind='galaxy'")}
    imgs = {r["id"]: r for r in con.execute(
        "SELECT id, source, target, filter, stats FROM images WHERE source IN ('jwst','hst')").fetchall()}
    rows = con.execute("""SELECT id, image_id, x, y, a, b, cnn_label, cnn_conf, cnn_probs FROM detections
                          WHERE cnn_probs IS NOT NULL ORDER BY id DESC LIMIT 6000""").fetchall()
    cand = []
    for d in rows:
        im = imgs.get(d["image_id"])
        key = f"{d['image_id']}:{d['id']}"
        if im is None or key in labelled or _solar(im["target"]):
            continue
        probs = json.loads(d["cnn_probs"])
        score, ent, margin = acquisition(probs)
        sus, why_sus = field_suspicion(im["target"], im["filter"])
        score += 0.35 * sus
        order = np.argsort(probs)[::-1]
        names = CLASSES_ALL[:len(probs)]
        if sus:
            reason = f"I may be wrong that this is a galaxy at all: {why_sus}."
        elif margin < 0.15:
            reason = f"I'm torn between {names[order[0]]} ({probs[order[0]]:.0%}) and {names[order[1]]} ({probs[order[1]]:.0%})."
        else:
            reason = f"My best guess, {names[order[0]]}, is only {probs[order[0]]:.0%} sure."
        st = json.loads(im["stats"] or "{}")
        w, h = st.get("size") or [1, 1]
        u, v = d["x"] / w, d["y"] / h
        r = min(0.08, max(0.012, d["a"] / w * 5))
        cand.append({"key": key, "image_id": d["image_id"], "det_id": d["id"], "score": round(score, 4),
                     "entropy": round(ent, 3), "margin": round(margin, 3), "reason": reason,
                     "top": [[names[i], round(float(probs[i]), 3)] for i in order[:3]],
                     "ai_label": d["cnn_label"], "target": im["target"], "source": im["source"],
                     "crop_url": f"/api/images/{d['image_id']}/crop?u={u:.4f}&v={v:.4f}&r={r:.4f}&size=384",
                     "u": round(u, 4), "v": round(v, 4), "r": round(r, 4)})
    # alternate the two kinds of question: "is this even a galaxy?" and "which galaxy type?"
    sus = sorted([c for c in cand if c["reason"].startswith("I may be wrong")], key=lambda c: -c["score"])
    clean = sorted([c for c in cand if not c["reason"].startswith("I may be wrong")], key=lambda c: -c["score"])
    cand = [c for pair in zip(clean, sus) for c in pair] + clean[len(sus):] + sus[len(clean):]
    out, per = [], {}
    for c in cand:                                          # diversity: a few per image
        if per.get(c["image_id"], 0) >= per_image:
            continue
        per[c["image_id"]] = per.get(c["image_id"], 0) + 1
        out.append(c)
        if len(out) >= n:
            break
    return out


# ---------- 2. LEARN: the label pool ----------

class _Images:
    """Full-resolution greyscale images (science data when we have it), cached while building the pool."""
    def __init__(self, con):
        self.con, self.cache = con, {}

    def get(self, image_id):
        if image_id not in self.cache:                     # keep one image in memory (queries are grouped by image)
            r = self.con.execute("SELECT path, stats FROM images WHERE id=?", (image_id,)).fetchone()
            sci = DATA / "science" / f"{image_id}.png"
            p = sci if sci.exists() else Path(r["path"]) if Path(r["path"]).is_absolute() else ROOT / r["path"]
            g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            self.cache = {image_id: (g, json.loads(r["stats"] or "{}").get("scale") or 1.0)}
        return self.cache[image_id]


def build_pool(con, size=SIZE, cap_star=700, cap_gas=600, cap_pseudo=2500, cap_archive=600):
    """-> list of dicts {x (uint8), fine, group, w, kind, holdout}. fine/group = -1 when unknown."""
    from .explain_lite import concentration, field_suspicion
    images = _Images(con)
    meta = {r["id"]: r for r in con.execute("SELECT id, target, filter FROM images WHERE source IN ('jwst','hst')").fetchall()}
    pool = []

    def add(image_id, d, fine, group, w, kind, key):
        g, sc = images.get(image_id)
        if g is None:
            return
        x = cutout(g, d["x"] / sc, d["y"] / sc, max(d["a"] / sc, 1.0), size=size)
        pool.append({"x": x, "fine": fine, "group": group, "w": w, "kind": kind, "holdout": _holdout(key)})

    # your labels
    rows = con.execute("SELECT key, label FROM user_labels WHERE kind='galaxy'").fetchall()
    for r in sorted(rows, key=lambda r: r["key"]):
        try:
            image_id, det_id = (int(v) for v in r["key"].split(":"))
        except ValueError:
            continue
        lab = LEGACY.get(r["label"])
        if lab is None or image_id not in meta:
            continue
        d = con.execute("SELECT x, y, a FROM detections WHERE id=?", (det_id,)).fetchone()
        if not d:
            continue
        fine, group = TEACH_LABELS[lab]
        add(image_id, d, -1 if fine is None else fine, GROUPS.index(group) if group else -1, 3.0, "user", r["key"])
    n_user = len(pool)

    # archive labels (cross-survey engine): Gaia parallax / SIMBAD / SDSS say star or galaxy; they outrank the detector
    archive = {}
    try:
        for r in con.execute("SELECT ref, verdict, verdict_conf FROM astronomical_objects WHERE kind='detection' AND verdict IN "
                             "('catalog_star', 'catalog_galaxy') AND verdict_conf >= 0.6"):
            archive[int(r["ref"].split(":")[2])] = r["verdict"]
    except Exception:  # noqa: BLE001  (table missing on old databases)
        pass
    per_arch, n_arch = {}, {"catalog_star": 0, "catalog_galaxy": 0}
    if archive:
        q = ",".join(str(i) for i in archive)
        for d in con.execute(f"SELECT image_id, id, x, y, a FROM detections WHERE id IN ({q}) ORDER BY image_id, score DESC"):
            v = archive[d["id"]]
            if d["image_id"] not in meta or n_arch[v] >= cap_archive or per_arch.get((d["image_id"], v), 0) >= 40:
                continue
            per_arch[(d["image_id"], v)] = per_arch.get((d["image_id"], v), 0) + 1
            n_arch[v] += 1
            # a galaxy of unknown type: group -2 = "any galaxy class" (loss pushes P(not a galaxy) down)
            add(d["image_id"], d, NG if v == "catalog_star" else -1, -1 if v == "catalog_star" else -2, 1.0,
                "archive_star" if v == "catalog_star" else "archive_galaxy", f"a{d['id']}")

    # weak "Not a galaxy": bright stars the detector is sure about, and flat patches in nebula fields
    stars = con.execute("""SELECT d.image_id, d.id, d.x, d.y, d.a FROM detections d JOIN images i ON i.id = d.image_id
                           WHERE i.source IN ('jwst','hst') AND d.label = 'star' ORDER BY d.image_id, d.score DESC""").fetchall()
    per = {}
    for d in stars:
        if sum(1 for p in pool if p["kind"] == "star") >= cap_star:
            break
        if per.get(d["image_id"], 0) >= 20 or _solar(meta.get(d["image_id"], {"target": ""})["target"]):
            continue
        if d["id"] in archive:            # the archives already decided (and may say it is a galaxy)
            continue
        per[d["image_id"]] = per.get(d["image_id"], 0) + 1
        add(d["image_id"], d, NG, -1, 0.4, "star", f"s{d['id']}")
    gal = con.execute("""SELECT d.image_id, d.id, d.x, d.y, d.a, d.b, d.theta, d.cnn_label, d.cnn_conf FROM detections d
                         JOIN images i ON i.id = d.image_id WHERE i.source IN ('jwst','hst') AND d.label LIKE '%galaxy%'
                         ORDER BY d.image_id, d.score DESC""").fetchall()
    n_gas = n_pseudo = 0
    per_gas, per_pseudo = {}, {}
    for d in gal:
        m = meta.get(d["image_id"])
        if m is None or _solar(m["target"]):
            continue
        if per_gas.get(d["image_id"], 0) >= 60 and per_pseudo.get(d["image_id"], 0) >= 120:
            continue
        sus, _ = field_suspicion(m["target"], m["filter"])
        g, sc = images.get(d["image_id"])
        if g is None:
            continue
        c = concentration(g, d["x"] / sc, d["y"] / sc, max(d["a"] / sc, 0.5), max(d["b"] / sc, 0.5), d["theta"] or 0)
        if sus and c is not None and c < 2.5 and n_gas < cap_gas and per_gas.get(d["image_id"], 0) < 60:
            add(d["image_id"], d, NG, -1, 0.5, "gas", f"g{d['id']}")
            n_gas += 1
            per_gas[d["image_id"]] = per_gas.get(d["image_id"], 0) + 1
        elif (not sus and c is not None and c > 3 and (d["cnn_conf"] or 0) >= 0.8 and d["cnn_label"] in CLASSES
              and n_pseudo < cap_pseudo and per_pseudo.get(d["image_id"], 0) < 120):
            # real galaxies from our own telescopes: keeps "not a galaxy" from meaning "looks like JWST"
            add(d["image_id"], d, CLASSES.index(d["cnn_label"]), -1, 0.7, "pseudo", f"p{d['id']}")
            n_pseudo += 1
            per_pseudo[d["image_id"]] = per_pseudo.get(d["image_id"], 0) + 1
    log.info("pool: %d user, %d archive stars, %d archive galaxies, %d stars, %d gas, %d pseudo-galaxies", n_user,
             n_arch["catalog_star"], n_arch["catalog_galaxy"], sum(1 for p in pool if p["kind"] == "star"), n_gas, n_pseudo)
    return pool


# ---------- 2b. fine-tune + 3. gate ----------

def _model(n_classes):
    import timm
    return timm.create_model("convnext_nano", pretrained=False, num_classes=n_classes, in_chans=1,
                             drop_rate=0.25, drop_path_rate=0.1)


def _load_champion():
    """Current champion as an 11-class model (a 10-class Galaxy10 model gets a fresh 11th output)."""
    import torch
    m = _model(11)
    live = MODELS / f"{LIVE}.pt"
    src = live if live.exists() else MODELS / "galaxy_cnn_v2.pt"
    sd = torch.load(src, map_location="cpu")
    if sd["head.fc.weight"].shape[0] == 10:
        own = m.state_dict()
        w, b = own["head.fc.weight"].clone(), own["head.fc.bias"].clone()
        w[:10], b[:10] = sd["head.fc.weight"], sd["head.fc.bias"]
        b[10] = -2.0                                            # start rarely predicting the new class
        sd = {**sd, "head.fc.weight": w, "head.fc.bias": b}
    m.load_state_dict(sd)
    return m, src.stem, sd


def _probs(model, x_u8, dev, tta=True):
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x_u8), 128):
            xb = torch.from_numpy(x_u8[i:i + 128].astype(np.float32) / 255.0)[:, None].to(dev)
            with torch.autocast("cuda", enabled=dev.type == "cuda"):
                ks = range(4) if tta else range(1)
                p = sum(model(torch.rot90(xb, k, (2, 3))).float().softmax(1) for k in ks) / len(ks)
            out.append(p.cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 11), np.float32)


def evaluate(model, dev, g10_val, hold, ten_class=False):
    """Metrics the gate compares. ten_class: the old champion can never answer 'Not a galaxy'."""
    xg, yg = g10_val
    pg = _probs(model, xg, dev)
    if ten_class:
        pg[:, NG] = 0
    m = {"g10_val_acc": round(float((pg.argmax(1) == yg).mean()), 4)}
    user = [h for h in hold if h["kind"] == "user"]
    if len(user) >= 5:
        pu = _probs(model, np.stack([h["x"] for h in user]), dev)
        if ten_class:
            pu[:, NG] = 0
        grp = np.zeros((11, len(GROUPS)))
        for gi, g in enumerate(GROUPS[:-1]):
            grp[COARSE[g], gi] = 1
        grp[NG, -1] = 1
        pred_g = (pu @ grp).argmax(1)
        true_g = np.array([h["group"] if h["group"] >= 0 else (len(GROUPS) - 1 if h["fine"] == NG else
                           next(gi for gi, g in enumerate(GROUPS[:-1]) if h["fine"] in COARSE[g])) for h in user])
        m["user_holdout_acc"] = round(float((pred_g == true_g).mean()), 4)
        m["user_holdout_n"] = len(user)
    ng = [h for h in hold if h["kind"] in ("star", "gas")]
    ps = [h for h in hold if h["kind"] == "pseudo"]
    if ng:
        p = _probs(model, np.stack([h["x"] for h in ng]), dev)
        m["not_galaxy_recall"] = 0.0 if ten_class else round(float((p.argmax(1) == NG).mean()), 4)
    if ps:
        p = _probs(model, np.stack([h["x"] for h in ps]), dev)
        m["galaxy_false_positive"] = 0.0 if ten_class else round(float((p.argmax(1) == NG).mean()), 4)
    ar = [h for h in hold if h["kind"] in ("archive_star", "archive_galaxy")]
    if ar and not ten_class:              # independent truth: Gaia / SIMBAD / SDSS, not our own detector
        p = _probs(model, np.stack([h["x"] for h in ar]), dev)
        truth = np.array([h["kind"] == "archive_star" for h in ar])
        m["archive_agreement"] = round(float(((p.argmax(1) == NG) == truth).mean()), 4)
        m["archive_n"] = len(ar)
    return m


def gate(champ, chall):
    reasons = []
    if chall["g10_val_acc"] < champ["g10_val_acc"] - 0.015:
        reasons.append(f"Galaxy10 accuracy fell {champ['g10_val_acc']:.3f} -> {chall['g10_val_acc']:.3f}")
    if "user_holdout_acc" in chall and chall["user_holdout_acc"] < champ.get("user_holdout_acc", 0) - 0.02:
        reasons.append(f"accuracy on your held-out labels fell {champ['user_holdout_acc']:.3f} -> {chall['user_holdout_acc']:.3f}")
    if "not_galaxy_recall" in chall and chall["not_galaxy_recall"] < max(0.6, champ.get("not_galaxy_recall", 0) - 0.02):
        reasons.append(f"not-a-galaxy recall too low ({chall['not_galaxy_recall']:.3f})")
    if "galaxy_false_positive" in chall and chall["galaxy_false_positive"] > max(0.1, champ.get("galaxy_false_positive", 0) + 0.02):
        reasons.append(f"too many real galaxies called 'not a galaxy' ({chall['galaxy_false_positive']:.3f})")
    if "archive_agreement" in chall and "archive_agreement" in champ and chall["archive_agreement"] < champ["archive_agreement"] - 0.03:
        reasons.append(f"agreement with the archives fell {champ['archive_agreement']:.3f} -> {chall['archive_agreement']:.3f}")
    return not reasons, reasons


def train_round(epochs=3, g10_per_epoch=6000, batch=64):
    """One full ask-learn-gate-swap round. Returns the registry entry."""
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    from .train import augment, load_galaxy10

    t0 = time.time()
    con = db.connect()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, y = load_galaxy10(SIZE)
    rng = np.random.default_rng(0)
    val_idx = np.concatenate([rng.permutation(np.flatnonzero(y == c))[: max(1, int(0.15 * (y == c).sum()))] for c in range(10)])
    tr_mask = np.ones(len(y), bool)
    tr_mask[val_idx] = False
    tr_idx = np.flatnonzero(tr_mask)
    pool = build_pool(con)
    train_pool = [p for p in pool if not p["holdout"]]
    hold = [p for p in pool if p["holdout"]]
    n_user = con.execute("SELECT COUNT(*) FROM user_labels WHERE kind='galaxy'").fetchone()[0]

    champion, champ_name, champ_sd = _load_champion()
    champion.to(dev)
    ten = champ_name != LIVE
    champ_m = evaluate(champion, dev, (x[val_idx], y[val_idx]), hold, ten_class=ten)
    log.info("champion %s: %s", champ_name, champ_m)

    model = _model(11)
    model.load_state_dict(champ_sd)
    model.to(dev)

    class Mix(Dataset):
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            img, fine, group, w = self.items[i]
            a = augment(img, np.random.default_rng())
            return torch.from_numpy(a.astype(np.float32))[None], fine, group, w

    grp = torch.zeros(len(GROUPS), 11)
    for gi, g in enumerate(GROUPS[:-1]):
        grp[gi, COARSE[g]] = 1
    grp[-1, NG] = 1
    grp = grp.to(dev)
    head = [p for n, p in model.named_parameters() if n.startswith("head")]
    body = [p for n, p in model.named_parameters() if not n.startswith("head")]
    opt = torch.optim.AdamW([{"params": body, "lr": 6e-5}, {"params": head, "lr": 4e-4}], weight_decay=0.05)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    for ep in range(epochs):
        g10 = rng.choice(tr_idx, size=min(g10_per_epoch, len(tr_idx)), replace=False)
        items = [(x[i], int(y[i]), -1, 1.0) for i in g10] + [(p["x"], p["fine"], p["group"], p["w"]) for p in train_pool]
        dl = DataLoader(Mix(items), batch_size=batch, shuffle=True, num_workers=0, drop_last=True)
        model.train()
        run, nb = 0.0, 0
        for xb, fine, group, w in dl:
            xb, fine, group, w = xb.to(dev), fine.to(dev), group.to(dev), w.float().to(dev)
            with torch.autocast("cuda", enabled=dev.type == "cuda"):
                logits = model(xb)
            logits = logits.float()
            loss = logits.new_zeros(())
            f = fine >= 0
            if f.any():
                loss = loss + (F.cross_entropy(logits[f], fine[f], reduction="none", label_smoothing=0.05) * w[f]).mean()
            g = (~f) & (group >= 0)
            if g.any():
                pg = logits[g].softmax(1) @ grp.T
                loss = loss + (F.nll_loss(torch.log(pg.clamp_min(1e-6)), group[g], reduction="none") * w[g]).mean()
            a = (~f) & (group == -2)
            if a.any():                     # archive says "a galaxy", type unknown
                p_ng = logits[a].softmax(1)[:, NG]
                loss = loss + (-torch.log((1 - p_ng).clamp_min(1e-6)) * w[a]).mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run += loss.item()
            nb += 1
        log.info("epoch %d/%d loss %.3f", ep + 1, epochs, run / max(nb, 1))
    chall_m = evaluate(model, dev, (x[val_idx], y[val_idx]), hold)
    ok, reasons = gate(champ_m, chall_m)
    reg = registry()
    version = f"galaxy_cnn_v2.{len([r for r in reg if r.get('promoted')]) + 1}"
    entry = {"round": len(reg) + 1, "time": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()), "champion": champ_name,
             "version": version if ok else None, "promoted": ok, "reasons": reasons,
             "champion_metrics": champ_m, "challenger_metrics": chall_m, "n_user_labels": n_user,
             "pool": {k: sum(1 for p in pool if p["kind"] == k) for k in ("user", "star", "gas", "pseudo")},
             "minutes": round((time.time() - t0) / 60, 1)}
    if ok:
        model.eval().float().cpu()
        sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
        torch.save(sd, MODELS / f"{LIVE}.pt")
        tmp = MODELS / f"{LIVE}.tmp.onnx"
        torch.onnx.export(model, torch.zeros(1, 1, SIZE, SIZE), tmp, input_names=["image"], output_names=["logits"],
                          dynamic_axes={"image": {0: "n"}, "logits": {0: "n"}}, opset_version=17, dynamo=False)
        card = {"version": version, "architecture": "Zoobot greyscale ConvNeXt-nano, fine-tuned by active learning",
                "size": SIZE, "channels": 1, "tta_rotations": 4, "normalize": {"mean": 0.0, "std": 1.0},
                "classes": CLASSES_ALL, "val_metrics": chall_m, "trained_at": entry["time"], "round": entry["round"],
                "data": entry["pool"], "n_user_labels": n_user}
        _write(MODELS / f"{LIVE}.json", card)
        tmp.replace(MODELS / f"{LIVE}.onnx")
        log.info("promoted %s", version)
    else:
        log.info("challenger rejected: %s", "; ".join(reasons))
    _write(REGISTRY, reg + [entry])
    if ok:
        reclassify(con)
    return entry


def reclassify(con, n=25):
    """Run the (new) live model over the newest deep-field images."""
    from . import infer
    infer.reload()
    ids = [r[0] for r in con.execute("SELECT id FROM images WHERE source IN ('jwst','hst') ORDER BY id DESC LIMIT ?", (n,))]
    for iid in ids:
        try:
            infer.classify_image(con, iid)
        except Exception as e:  # noqa: BLE001
            log.warning("re-classify image %s failed: %s", iid, e)
    log.info("re-classified %d images with %s", len(ids), infer.version())


# ---------- the pipeline hook ----------

def _alive(pid):
    if not pid:
        return False
    try:
        if os.name == "nt":
            out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True, timeout=10).stdout
            return str(pid) in out
        os.kill(pid, 0)
        return True
    except Exception:  # noqa: BLE001
        return False


def training_state():
    st = _read(STATE, {})
    if st.get("running") and not _alive(st.get("pid")):
        st["running"] = False
    return st


def status(con):
    reg = registry()
    n_user = con.execute("SELECT COUNT(*) FROM user_labels WHERE kind='galaxy'").fetchone()[0]
    last = next((r for r in reversed(reg)), None)
    since = n_user - (last["n_user_labels"] if last else 0)
    from . import infer
    card = infer.card() if infer.available() else {}
    return {"model": card.get("version"), "classes": len(card.get("classes") or []), "live": card.get("version", "").startswith("galaxy_cnn_v2."),
            "user_labels": n_user, "new_since_last_round": since, "threshold": MIN_NEW_LABELS,
            "training": training_state(), "rounds": reg[-12:], "gpu": _gpu()}


def _gpu():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def maybe_retrain(con, force=False):
    """Start a background training round if it is worth it. Returns a short status string."""
    if training_state().get("running"):
        return "training already running"
    if not _gpu():
        return "no GPU here: active learning trains only where PyTorch + CUDA are available"
    reg = registry()
    n_user = con.execute("SELECT COUNT(*) FROM user_labels WHERE kind='galaxy'").fetchone()[0]
    last = reg[-1] if reg else None
    hours = (time.time() - time.mktime(time.strptime(last["time"], "%Y-%m-%d %H:%M:%S UTC"))) / 3600 + time.timezone / 3600 if last else 1e9
    newer = con.execute("SELECT COUNT(*) FROM images WHERE source IN ('jwst','hst') AND fetched_at > ?",
                        (last["time"].replace(" UTC", "") if last else "",)).fetchone()[0]
    why = ("forced" if force else "first round" if not last
           else f"{n_user - last['n_user_labels']} new labels" if n_user - last["n_user_labels"] >= MIN_NEW_LABELS
           else "daily refresh with new images" if hours >= REFRESH_HOURS and newer else None)
    if not why:
        return f"waiting: {n_user - (last['n_user_labels'] if last else 0)}/{MIN_NEW_LABELS} new labels"
    MODELS.mkdir(parents=True, exist_ok=True)
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    with open(TRAIN_LOG, "a", encoding="utf-8") as f:
        f.write(f"\n=== round started {time.strftime('%Y-%m-%d %H:%M:%S')} ({why}) ===\n")
        p = subprocess.Popen([sys.executable, "-m", "cosmic_watch.ml.active", "train"], cwd=str(ROOT),
                             stdout=f, stderr=subprocess.STDOUT, creationflags=flags,
                             env={**os.environ, "PYTHONUTF8": "1"})
    _write(STATE, {"running": True, "pid": p.pid, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "why": why})
    return f"training started ({why})"


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "queue", "status"])
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    con = db.connect()
    if args.cmd == "queue":
        for c in queue(con, 15):
            print(f"{c['score']:.3f}  {c['target'][:22]:22s}  {c['top'][0][0]:24s} {c['reason']}")
    elif args.cmd == "status":
        print(json.dumps(status(con), indent=2))
    else:
        try:
            e = train_round()
            print(json.dumps({k: e[k] for k in ("round", "promoted", "version", "reasons", "champion_metrics", "challenger_metrics")}, indent=2))
        finally:
            st = _read(STATE, {})
            st.update(running=False, finished=time.strftime("%Y-%m-%d %H:%M:%S"))
            _write(STATE, st)


if __name__ == "__main__":
    main()
