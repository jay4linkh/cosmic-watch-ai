"""Computer vision on Mars rover images.

* Scene classifier (our ConvNeXt, trained on NASA's labelled Curiosity images: drill hole, layers,
  light-toned veins, sand, wheel tracks, close-up rock, distant landscape, rover parts, ...)
* Horizon / sky: smooth, bright regions connected to the top edge (slanted skylines too)
* Roughness: edge density of the terrain (rocky vs sandy), and the mean colour
"""
import json

import cv2
import numpy as np

from ..config import ROOT

MODEL = ROOT / "models" / "mars_cnn_v1.onnx"
CARD = ROOT / "models" / "mars_cnn_v1.json"
# classes worth flagging to the agent (geology, sky) vs rover engineering
SCIENCE = {"Drill hole", "Layers", "Light-toned veins", "Float", "Night sky", "Sun", "DRT spot"}
NOTES = {
    "Arm cover": "a cover on the robotic arm's instruments",
    "Other rover part": "part of the rover itself",
    "Artifact": "an image artifact (calibration, blank or damaged frame)",
    "Nearby surface": "the ground right next to the rover",
    "Close-up rock": "a close-up of a rock's surface texture",
    "DRT": "the Dust Removal Tool, which brushes rock surfaces clean",
    "DRT spot": "a rock patch brushed clean by the Dust Removal Tool",
    "Distant landscape": "the distant landscape: crater rim, mountains, horizon",
    "Drill hole": "a hole drilled by the rover to collect rock powder for analysis",
    "Night sky": "the Martian night sky",
    "Float": "a loose rock ('float') that rolled or was carried from elsewhere",
    "Layers": "layered rock: sediments laid down over time, often by ancient water",
    "Light-toned veins": "light-toned veins: minerals (often calcium sulphate) left by water flowing through cracks",
    "Mastcam cal target": "the camera's colour calibration target",
    "Sand": "sand or dust",
    "Sun": "the Sun, imaged to measure dust in the atmosphere",
    "Wheel": "one of the rover's wheels",
    "Wheel joint": "a wheel joint or suspension part",
    "Wheel tracks": "the rover's own wheel tracks",
}
_sess = None


def available():
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return MODEL.exists() and CARD.exists()


def _session():
    global _sess
    if _sess is None:
        import onnxruntime as ort
        o = ort.SessionOptions()
        o.intra_op_num_threads = 2
        _sess = (ort.InferenceSession(str(MODEL), o, providers=["CPUExecutionProvider"]),
                 json.loads(CARD.read_text(encoding="utf-8")))
    return _sess


def classify(rgb):
    """RGB uint8 -> top-3 [(class, p)], averaged over the image and its mirror."""
    s, card = _session()
    size = card["size"]
    x = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    x = (x - np.array(card["mean"], np.float32)) / np.array(card["std"], np.float32)
    batch = np.stack([x.transpose(2, 0, 1), x[:, ::-1].transpose(2, 0, 1)]).astype(np.float32)
    logits = s.run(None, {"image": np.ascontiguousarray(batch)})[0]
    e = np.exp(logits - logits.max(1, keepdims=True))
    p = (e / e.sum(1, keepdims=True)).mean(0)
    order = np.argsort(-p)[:3]
    return [[card["classes"][int(i)], round(float(p[i]), 3)] for i in order]


def horizon(gray):
    """Fraction of the image that is sky: smooth, bright pixels connected to the top edge (works for slanted skylines)."""
    g = cv2.resize(gray, (320, max(1, int(320 * gray.shape[0] / gray.shape[1]))), interpolation=cv2.INTER_AREA).astype(np.float32)
    mu = cv2.blur(g, (7, 7))
    sd = np.sqrt(np.maximum(cv2.blur(g * g, (7, 7)) - mu * mu, 0))
    lower = slice(g.shape[0] // 2, None)
    smooth = sd < max(2.0, 0.35 * float(np.median(sd[lower])))
    bright = mu > np.percentile(g[lower], 60)
    m = cv2.morphologyEx((smooth & bright).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    _, lab = cv2.connectedComponents(m)
    top = [v for v in np.unique(lab[0]) if v]
    return round(float(np.isin(lab, top).mean()), 3) if top else 0.0


def analyse(path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    if max(bgr.shape[:2]) > 1600:
        f = 1600 / max(bgr.shape[:2])
        bgr = cv2.resize(bgr, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (0, 0), 1.2), 60, 140)
    sky = horizon(gray)
    terrain = edges[int(sky * gray.shape[0]):]
    mean = rgb.reshape(-1, 3).mean(0)
    out = {"size": [int(bgr.shape[1]), int(bgr.shape[0])], "sky_fraction": sky,
           "roughness": round(float((terrain > 0).mean()) if terrain.size else 0.0, 4),
           "colour": [int(v) for v in mean], "greyscale": bool(np.ptp(mean) < 4)}
    if available():
        top = classify(rgb)
        out.update(scene=top[0][0], scene_conf=top[0][1], scene_top=top, scene_note=NOTES.get(top[0][0]))
    return out
