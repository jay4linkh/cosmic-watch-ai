"""Where things are in a rover image, not just what the whole image is.

* regions(): the scene CNN is slid over the image at two scales; where neighbouring windows agree on a geology or
  hardware class, that area is outlined. The CNN was trained on whole Curiosity images, so a window is a harder
  input than it has seen: only confident, repeated answers are kept, and they are reported as "the CNN sees".
* rocks(): rock candidates on the ground from local contrast (a sunlit face) and, when the Sun is low enough, the
  shadow next to it. Sizes are in pixels: without range data they cannot be turned into centimetres.
Neither has ground truth here; both are candidates for a person to look at, and the viewer says so.
"""
import cv2
import numpy as np

from . import mars as scene

# classes that mean something when found in part of an image (whole-image classes like "Nearby surface" do not)
LOCAL = {"Layers", "Light-toned veins", "Float", "Drill hole", "DRT spot", "Wheel tracks", "Sand", "Wheel", "Other rover part", "Mastcam cal target"}
MIN_P, MIN_HEAT = 0.75, 0.7
# Rocks are only looked for where the scene CNN says the image shows loose rocks or sand. Tested on real rover images:
# on fractured bedrock, in close-ups and in far views this method marks shadows, cracks and holes, so it is not run there.
ROCK_SCENES = {"Float", "Sand"}


def _probs(tiles):
    """Batch of RGB uint8 tiles -> class probabilities."""
    s, card = scene._session()
    size, mean, std = card["size"], np.array(card["mean"], np.float32), np.array(card["std"], np.float32)
    x = np.stack([((cv2.resize(t, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0 - mean) / std).transpose(2, 0, 1) for t in tiles])
    out = []
    for i in range(0, len(x), 8):
        logits = s.run(None, {"image": np.ascontiguousarray(x[i:i + 8], dtype=np.float32)})[0]
        e = np.exp(logits - logits.max(1, keepdims=True))
        out.append(e / e.sum(1, keepdims=True))
    return np.concatenate(out), card["classes"]


def regions(rgb, max_regions=6):
    """Areas where the scene CNN, looking through overlapping windows, keeps seeing the same thing."""
    if not scene.available():
        return []
    h, w = rgb.shape[:2]
    if min(h, w) < 300:
        return []
    boxes = []
    for frac in (0.5, 0.34):                                # windows of half and a third of the frame, half-overlapping
        tw, th = int(w * frac), int(h * frac)
        for y in np.unique(np.linspace(0, h - th, max(1, round(2 / frac) - 1)).astype(int)):
            for x in np.unique(np.linspace(0, w - tw, max(1, round(2 / frac) - 1)).astype(int)):
                boxes.append((x, y, tw, th))
    p, classes = _probs([rgb[y:y + th, x:x + tw] for x, y, tw, th in boxes])
    out = []
    for ci, name in enumerate(classes):
        if name not in LOCAL:
            continue
        heat, cover = np.zeros((h // 8, w // 8), np.float32), np.zeros((h // 8, w // 8), np.float32)
        for (x, y, tw, th), pr in zip(boxes, p):
            sl = (slice(y // 8, (y + th) // 8), slice(x // 8, (x + tw) // 8))
            cover[sl] += 1
            if pr[ci] >= MIN_P and pr.argmax() == ci:
                heat[sl] += pr[ci]
        heat /= np.maximum(cover, 1)                        # mean vote of every window that covers this spot
        n, lab, st, cen = cv2.connectedComponentsWithStats((heat >= MIN_HEAT).astype(np.uint8), connectivity=8)
        for i in range(1, n):
            if st[i, cv2.CC_STAT_AREA] < 0.02 * heat.size:
                continue
            bx, by, bw, bh = (int(st[i, k]) * 8 for k in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
            out.append({"label": f"CNN: {name}", "cls": name, "x": bx + bw / 2, "y": by + bh / 2, "a": bw / 2, "b": bh / 2, "theta": 0.0,
                        "score": round(float(heat[lab == i].mean()), 3), "note": scene.NOTES.get(name)})
    return sorted(out, key=lambda d: -d["score"])[:max_regions]


def sky_mask(gray):
    """Smooth, bright pixels connected to the top edge (same rule as mars.horizon), as a full-size mask."""
    hs = max(1, int(320 * gray.shape[0] / gray.shape[1]))
    g = cv2.resize(gray, (320, hs), interpolation=cv2.INTER_AREA).astype(np.float32)
    mu = cv2.blur(g, (7, 7))
    sd = np.sqrt(np.maximum(cv2.blur(g * g, (7, 7)) - mu * mu, 0))
    lower = slice(g.shape[0] // 2, None)
    m = ((sd < max(2.0, 0.35 * float(np.median(sd[lower])))) & (mu > np.percentile(g[lower], 60))).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    _, lab = cv2.connectedComponents(m)
    top = [v for v in np.unique(lab[0]) if v]
    sky = np.isin(lab, top).astype(np.uint8) if top else np.zeros_like(m)
    return cv2.resize(cv2.dilate(sky, np.ones((5, 5), np.uint8)), (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)


MAX_BUSY = 0.30               # if more of the ground than this "stands out", it is bedrock or a close-up, not rocks on soil


def rocks(gray, exclude=None, max_rocks=300):
    """Loose rocks on smooth ground: patches that differ from the ground around them in brightness or texture.

    Works where the ground is sand or soil. On fractured bedrock, in close-ups and in rover-shadow scenes most of the
    frame "stands out", there is no smooth ground to compare against, and the function says so instead of guessing.
    """
    h, w = gray.shape
    f = 640 / max(h, w)                                       # work at a fixed scale: thresholds mean the same in every camera
    g = cv2.GaussianBlur(cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_AREA), (0, 0), 1.0)
    hh, ww = g.shape
    ground = cv2.resize((~sky_mask(gray)).astype(np.uint8), (ww, hh), interpolation=cv2.INTER_NEAREST).astype(bool) & (g > 4)
    if exclude is not None:
        ground &= ~cv2.resize(exclude.astype(np.uint8), (ww, hh), interpolation=cv2.INTER_NEAREST).astype(bool)
    stats = {"n_rocks": 0, "ground_fraction": round(float(ground.mean()), 3), "image_width_px": int(w)}
    if ground.mean() < 0.15:
        return [], {**stats, "skipped": "almost no ground in view"}
    base = cv2.medianBlur(g, 61)                              # the ground with everything rock-sized removed
    dev = g.astype(np.float32) - base
    gf = g.astype(np.float32)
    tex = np.sqrt(np.maximum(cv2.blur(gf * gf, (7, 7)) - cv2.blur(gf, (7, 7)) ** 2, 0))
    mad = 1.4826 * float(np.median(np.abs(dev[ground] - np.median(dev[ground])))) + 1.0
    tmed = float(np.median(tex[ground])) + 0.5
    stand = ((np.abs(dev) > 4.0 * mad) | (tex > 3.5 * tmed)) & ground
    e = max(4, int(0.012 * max(hh, ww)))                     # frame edges and borders stand out too: ignore them
    stand[:e], stand[-e:], stand[:, :e], stand[:, -e:] = False, False, False, False
    busy = float(stand.sum()) / max(float(ground.sum()), 1)
    stats["standing_out_fraction"] = round(busy, 3)
    if busy > MAX_BUSY:
        return [], {**stats, "skipped": "no smooth ground to compare against (bedrock, close-up or heavy shadow)"}
    m = cv2.morphologyEx(stand.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cs:
        area = cv2.contourArea(c)
        if area < 40 or area > 0.2 * hh * ww or len(c) < 5:
            continue
        hull = cv2.contourArea(cv2.convexHull(c))
        (cx, cy), (d1, d2), ang = cv2.fitEllipse(c)
        if hull <= 0 or area / hull < 0.6 or max(d1, d2) / max(min(d1, d2), 1e-6) > 3.5:           # ragged or thread-like
            continue
        mask = np.zeros((hh, ww), np.uint8)
        cv2.drawContours(mask, [c], -1, 1, -1)
        mb = mask.astype(bool)
        lit = float(np.percentile(dev[mb], 80))
        # a rock shows a sunlit face (a third of it brighter than the ground) or is rough without being mostly dark.
        # Mostly-dark patches are shadows, sand-filled hollows and holes in bedrock. A smooth dark rock on bright dust
        # is lost by this rule: precision over recall.
        bright_f, dark_f = float((dev[mb] > 2.0 * mad).mean()), float((dev[mb] < -2.0 * mad).mean())
        if bright_f < 0.33 and not (dark_f < 0.35 and float(tex[mb].mean()) >= 2.5 * tmed):
            continue
        shadow = bool((dev[mb] < -4.0 * mad).mean() > 0.08)
        score = min(1.0, 0.45 + 0.1 * lit / mad / 4 + 0.15 * min(float(tex[mb].mean()) / tmed / 3, 1.5) + (0.15 if shadow else 0))
        out.append({"label": "rock", "x": float(cx / f), "y": float(cy / f), "a": float(max(d1, d2) / 2 / f), "b": float(min(d1, d2) / 2 / f),
                    "theta": float(np.radians(ang - 90)), "score": round(score, 3), "area": float(area / f / f), "shadow": shadow})
    out.sort(key=lambda d: -d["area"])
    out = out[:max_rocks]
    diam = np.array([2 * np.sqrt(d["area"] / np.pi) for d in out]) if out else np.array([])
    stats.update(n_rocks=len(out), rock_cover_pct=round(100 * float(sum(d["area"] for d in out)) * f * f / max(float(ground.sum()), 1), 2),
                 with_shadow=int(sum(d["shadow"] for d in out)),
                 diameter_px={"median": round(float(np.median(diam)), 1), "p90": round(float(np.percentile(diam, 90)), 1), "max": round(float(diam.max()), 1)} if len(diam) else None)
    return out, stats


def find(bgr, scene_name=None):
    """All located features for one image: (detections, stats)."""
    rgb, gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    regs = regions(rgb)
    stats = {"features": [{k: r[k] for k in ("cls", "score", "note")} for r in regs], "detector": "mars-2"}
    dets = list(regs)
    if scene_name in ROCK_SCENES:
        hard = np.zeros(gray.shape, bool)                    # do not call rover hardware a rock
        for r in regs:
            if r["cls"] in ("Wheel", "Other rover part", "Mastcam cal target"):
                hard[int(max(0, r["y"] - r["b"])):int(r["y"] + r["b"]), int(max(0, r["x"] - r["a"])):int(r["x"] + r["a"])] = True
        rk, rs = rocks(gray, hard)
        dets += rk
        stats["rocks"] = rs
    else:
        stats["rocks"] = {"n_rocks": 0, "skipped": "only run on scenes of loose rocks or sand"}
    return dets, stats
