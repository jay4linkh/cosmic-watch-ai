"""Mars rock candidates: found on smooth ground, quiet on empty ground, shadows and holes are not rocks,
and the detector is only run on scenes of loose rocks."""
import cv2
import numpy as np

from cosmic_watch.ml import mars_features as mf


def ground(seed=0, size=(600, 800)):
    rng = np.random.default_rng(seed)
    return cv2.GaussianBlur(110 + rng.normal(0, 4, size).astype(np.float32), (0, 0), 1.0)


def add_rock(img, x, y, r, seed=1):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:img.shape[0], :img.shape[1]]
    m = ((xx - x) / r) ** 2 + ((yy - y) / (0.8 * r)) ** 2 < 1
    img[m] = 185 + rng.normal(0, 14, int(m.sum()))                              # a bright, rough face
    img[((xx - x - 0.5 * r) / r) ** 2 + ((yy - y - 0.7 * r) / (0.5 * r)) ** 2 < 1] *= np.where(m, 1.0, 0.45)[((xx - x - 0.5 * r) / r) ** 2 + ((yy - y - 0.7 * r) / (0.5 * r)) ** 2 < 1]
    return img


def u8(img):
    return np.clip(img, 0, 255).astype(np.uint8)


def test_finds_rocks_of_several_sizes():
    img = ground()
    truth = [(200, 200, 60), (520, 180, 30), (400, 430, 14), (650, 450, 22)]
    for i, (x, y, r) in enumerate(truth):
        add_rock(img, x, y, r, i)
    rocks, st = mf.rocks(u8(img))
    for x, y, r in truth:
        d = min(np.hypot(k["x"] - x, k["y"] - y) for k in rocks)
        assert d < r, f"rock at {x},{y} (radius {r}) not found"
    assert st["n_rocks"] <= len(truth) + 2 and st["rock_cover_pct"] > 1
    big = max(rocks, key=lambda k: k["area"])
    assert abs(big["x"] - 200) < 40 and 80 < 2 * big["a"] < 220                 # the outline is about the size of the rock


def test_empty_ground_has_no_rocks():
    rocks, st = mf.rocks(u8(ground(3)))
    assert st["n_rocks"] == 0


def test_shadow_or_hole_alone_is_not_a_rock():
    img = ground(4)
    yy, xx = np.mgrid[:img.shape[0], :img.shape[1]]
    img[((xx - 300) / 50) ** 2 + ((yy - 300) / 35) ** 2 < 1] = 30                # a dark smooth patch
    rocks, _ = mf.rocks(u8(img))
    assert not rocks


def test_frame_border_is_ignored():
    img = ground(5)
    add_rock(img, 400, 300, 40)
    img[:3], img[-3:], img[:, :3], img[:, -3:] = 0, 0, 0, 0                      # a black frame once swallowed the whole image
    rocks, _ = mf.rocks(u8(img))
    assert any(np.hypot(k["x"] - 400, k["y"] - 300) < 40 for k in rocks)


def test_only_run_on_loose_rock_scenes(monkeypatch):
    monkeypatch.setattr(mf, "regions", lambda rgb: [])
    img = ground(6)
    add_rock(img, 400, 300, 40)
    bgr = cv2.cvtColor(u8(img), cv2.COLOR_GRAY2BGR)
    dets, st = mf.find(bgr, "Float")
    assert st["rocks"]["n_rocks"] >= 1 and any(d["label"] == "rock" for d in dets)
    dets, st = mf.find(bgr, "Nearby surface")
    assert not dets and "skipped" in st["rocks"]
