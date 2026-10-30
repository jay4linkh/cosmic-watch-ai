"""Run the galaxy CNN on live images (ONNX Runtime, CPU; no PyTorch needed).

    python -m cosmic_watch.ml.infer            # classify every stored JWST/Hubble image
    python -m cosmic_watch.ml.infer 12 13      # just these image ids
"""
import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np

from .. import db
from ..config import ROOT
from .classes import CLASSES, COARSE
from .cutouts import cutout, normalize

log = logging.getLogger("cosmic_watch")

MODEL = ROOT / "models" / "galaxy_cnn_v1.onnx"
MAX_PER_IMAGE = 120        # brightest galaxies per image
MIN_A_PX = 1.5             # skip sources too small to have a shape (full-res px)

_session = None


def available() -> bool:
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return MODEL.exists()


def session():
    global _session
    if _session is None:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        _session = ort.InferenceSession(str(MODEL), opts, providers=["CPUExecutionProvider"])
    return _session


def predict(cutouts_u8: np.ndarray) -> np.ndarray:
    """(N, H, W) uint8 -> (N, 10) probabilities."""
    out = []
    s = session()
    for i in range(0, len(cutouts_u8), 64):
        logits = s.run(None, {"image": normalize(cutouts_u8[i:i + 64])})[0]
        e = np.exp(logits - logits.max(1, keepdims=True))
        out.append(e / e.sum(1, keepdims=True))
    return np.concatenate(out) if out else np.zeros((0, len(CLASSES)), np.float32)


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
    cuts = np.stack([cutout(gray, d["x"] / scale, d["y"] / scale, d["a"] / scale) for d in dets])
    probs = predict(cuts)
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
        raise SystemExit(f"model not found: {MODEL} (train it with python -m cosmic_watch.ml.train)")
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
