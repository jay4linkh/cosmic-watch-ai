"""Run the galaxy CNN on live images (ONNX Runtime, CPU; no PyTorch needed).

    python -m cosmic_watch.ml.infer            # classify every stored JWST/Hubble image
    python -m cosmic_watch.ml.infer 12 13      # just these image ids

Uses the newest model in models/ (v2 Zoobot ConvNeXt if present, else v1 ResNet-18);
set COSMIC_CNN=galaxy_cnn_v1 to pin one. Input size, channels, normalisation and
rotation test-time augmentation all come from the model card next to the .onnx file.
"""
import argparse
import json
import logging
import os
from pathlib import Path

import cv2
import numpy as np

from .. import db
from ..config import ROOT
from .classes import CLASSES, COARSE
from .cutouts import MEAN, STD, SIZE, cutout, normalize

log = logging.getLogger("cosmic_watch")

MODEL_DIR = ROOT / "models"
VERSIONS = ["galaxy_cnn_v2", "galaxy_cnn_v1"]   # newest first
MAX_PER_IMAGE = 120        # brightest galaxies per image
MIN_A_PX = 1.5             # skip sources too small to have a shape (full-res px)

_session = None
_card = None


def _pick() -> str | None:
    want = os.environ.get("COSMIC_CNN")
    for v in ([want] if want else VERSIONS):
        if (MODEL_DIR / f"{v}.onnx").exists():
            return v
    return None


def available() -> bool:
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return _pick() is not None


def card() -> dict:
    """The active model's card, with defaults for v1 cards that predate the size fields."""
    global _card
    if _card is None:
        v = _pick()
        c = json.loads((MODEL_DIR / f"{v}.json").read_text(encoding="utf-8")) if v else {}
        c.setdefault("version", v)
        c.setdefault("size", SIZE)
        c.setdefault("channels", 3)
        c.setdefault("tta_rotations", 1)
        c.setdefault("normalize", {"mean": MEAN, "std": STD})
        _card = c
    return _card


def version() -> str | None:
    return card().get("version")


def session():
    global _session
    if _session is None:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        _session = ort.InferenceSession(str(MODEL_DIR / f"{version()}.onnx"), opts, providers=["CPUExecutionProvider"])
    return _session


def predict(cutouts_u8: np.ndarray) -> np.ndarray:
    """(N, size, size) uint8 -> (N, 10) probabilities, averaged over 90-degree rotations if the card says so."""
    c, s = card(), session()
    norm = c["normalize"]
    out = []
    for i in range(0, len(cutouts_u8), 32):
        x = normalize(cutouts_u8[i:i + 32], norm["mean"], norm["std"], c["channels"])
        acc = 0
        for k in range(c["tta_rotations"]):
            logits = s.run(None, {"image": np.ascontiguousarray(np.rot90(x, k, (2, 3)))})[0]
            e = np.exp(logits - logits.max(1, keepdims=True))
            acc = acc + e / e.sum(1, keepdims=True)
        out.append(acc / c["tta_rotations"])
    return np.concatenate(out) if out else np.zeros((0, len(CLASSES)), np.float32)


def classify_cutouts(gray: np.ndarray, sources) -> np.ndarray:
    """sources: iterable of (x, y, a) in gray's pixels -> (N, 10) probabilities."""
    size = card()["size"]
    cuts = [cutout(gray, x, y, a, size=size) for x, y, a in sources]
    return predict(np.stack(cuts)) if cuts else np.zeros((0, len(CLASSES)), np.float32)


def coarse(probs: np.ndarray) -> dict:
    return {k: float(probs[idx].sum()) for k, idx in COARSE.items()}


def classify_image(con, image_id: int) -> int:
    """Classify the galaxies in one stored image; writes cnn_* columns on detections."""
    img = con.execute("SELECT path, stats FROM images WHERE id=?", (image_id,)).fetchone()
    if not img:
        return 0
    p = Path(img["path"])
    p = p if p.is_absolute() else ROOT / p
    gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return 0
    scale = json.loads(img["stats"] or "{}").get("scale", 1.0) or 1.0
    dets = con.execute("""SELECT id, x, y, a FROM detections WHERE image_id=? AND label LIKE '%galaxy%'
                          ORDER BY score DESC LIMIT ?""", (image_id, MAX_PER_IMAGE)).fetchall()
    dets = [d for d in dets if d["a"] / scale >= MIN_A_PX]
    if not dets:
        return 0
    probs = classify_cutouts(gray, [(d["x"] / scale, d["y"] / scale, d["a"] / scale) for d in dets])
    rows = []
    for d, pr in zip(dets, probs):
        k = int(pr.argmax())
        rows.append((CLASSES[k], float(pr[k]), json.dumps([round(float(v), 4) for v in pr]), d["id"]))
    con.executemany("UPDATE detections SET cnn_label=?, cnn_conf=?, cnn_probs=? WHERE id=?", rows)
    con.commit()
    return len(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*", type=int)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not available():
        raise SystemExit(f"no model in {MODEL_DIR} (train one with python -m cosmic_watch.ml.train)")
    log.info("model %s (%s)", version(), card().get("architecture", "?"))
    con = db.connect()
    ids = args.ids or [r[0] for r in con.execute("SELECT id FROM images WHERE source IN ('jwst','hst') ORDER BY id")]
    total = 0
    for iid in ids:
        n = classify_image(con, iid)
        total += n
        log.info("image %s: %d galaxies classified", iid, n)
    summary = con.execute("""SELECT cnn_label, COUNT(*) n, ROUND(AVG(cnn_conf), 2) c FROM detections
                             WHERE cnn_label IS NOT NULL GROUP BY cnn_label ORDER BY n DESC""").fetchall()
    log.info("total %d · %s", total, ", ".join(f"{r['cnn_label']} {r['n']} (avg {r['c']})" for r in summary))


if __name__ == "__main__":
    main()
