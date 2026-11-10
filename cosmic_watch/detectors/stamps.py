"""Render alert cutouts and score the difference image.

An alert stamp has three panels: science (tonight), template (the reference sky)
and difference (science minus template). A real new event is a clean point of
light at the centre of the difference panel.
"""
import cv2
import numpy as np

PANEL = 150


def _stretch(a):
    """Robust asinh stretch to 0-255."""
    a = np.nan_to_num(a.astype(np.float32))
    lo, hi = np.percentile(a, [1, 99.7])
    if hi <= lo:
        hi = lo + 1
    x = np.clip((a - lo) / (hi - lo), 0, 1)
    x = np.arcsinh(10 * x) / np.arcsinh(10)
    return (x * 255).astype(np.uint8)


def diff_snr(diff) -> float:
    """Signal-to-noise of the central peak in the difference image (our 'real event' score)."""
    d = np.nan_to_num(diff.astype(np.float32))
    h, w = d.shape
    c = max(2, min(h, w) // 8)
    core = d[h // 2 - c:h // 2 + c + 1, w // 2 - c:w // 2 + c + 1]
    ring = np.ones_like(d, bool)
    ring[h // 2 - 2 * c:h // 2 + 2 * c + 1, w // 2 - 2 * c:w // 2 + 2 * c + 1] = False
    bg = d[ring]
    med = np.median(bg)
    noise = 1.4826 * np.median(np.abs(bg - med)) + 1e-6
    return float((np.abs(core - med).max()) / noise)


def render(stamps: dict, out_path, title: str = ""):
    panels = []
    for name in ("science", "template", "difference"):
        p = cv2.resize(_stretch(stamps[name]), (PANEL, PANEL), interpolation=cv2.INTER_NEAREST)
        p = cv2.cvtColor(p, cv2.COLOR_GRAY2BGR)
        cv2.putText(p, name, (4, PANEL - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
        panels.append(p)
    strip = np.hstack([np.pad(p, ((0, 0), (0, 2), (0, 0))) for p in panels])
    if title:
        bar = np.zeros((20, strip.shape[1], 3), np.uint8)
        cv2.putText(bar, title[:60], (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        strip = np.vstack([bar, strip])
    cv2.imwrite(str(out_path), strip)
    return out_path


# ---------- high-quality display renders for the zoom viewer ----------
HQ_VERSION = "hq2"
HQ_SIDE = 1536
VIEWS = ("change", "science", "template", "difference")
_DEEP = np.array([[0.0, 2, 3, 10], [0.22, 14, 24, 70], [0.5, 60, 110, 195], [0.78, 185, 212, 245], [1.0, 255, 250, 236]], np.float32)   # t, R, G, B


def sky_stats(a):
    """Median and robust noise of the outer part of a cutout (the centre holds the object)."""
    d = np.nan_to_num(a.astype(np.float32))
    h, w = d.shape
    c = max(2, min(h, w) // 4)
    ring = np.ones_like(d, bool)
    ring[h // 2 - c:h // 2 + c + 1, w // 2 - c:w // 2 + c + 1] = False
    bg = d[ring]
    med = float(np.median(bg))
    return med, float(1.4826 * np.median(np.abs(bg - med)) + 1e-9)


def _asinh01(a, beta_sigma=3.0, black_sigma=-1.5, white_pct=99.8):
    """Noise-aware asinh stretch to 0..1: the black point sits just under the sky, faint structure stays visible."""
    med, sig = sky_stats(a)
    x = np.nan_to_num(a.astype(np.float32)) - (med + black_sigma * sig)
    hi = max(float(np.percentile(x, white_pct)), 8 * sig)
    return np.clip(np.arcsinh(np.clip(x, 0, None) / (beta_sigma * sig)) / np.arcsinh(hi / (beta_sigma * sig)), 0, 1)


def _signed01(a, beta_sigma=5.0):
    """Difference image to -1..1 (0 = no change), asinh in units of its own noise."""
    med, sig = sky_stats(a)
    x = np.nan_to_num(a.astype(np.float32)) - med
    hi = max(float(np.percentile(np.abs(x), 99.8)), 8 * sig)
    return np.clip(np.arcsinh(x / (beta_sigma * sig)) / np.arcsinh(hi / (beta_sigma * sig)), -1, 1)


def _up(x, raw):
    k = max(1, HQ_SIDE // max(x.shape[:2]))
    size = (x.shape[1] * k, x.shape[0] * k)
    if raw:                                            # the real pixel grid, nothing between the pixels
        return cv2.resize(x, size, interpolation=cv2.INTER_NEAREST)
    return cv2.resize(x, size, interpolation=cv2.INTER_CUBIC)     # display interpolation only (applied after the stretch)


def _deep(t):
    t = np.clip(t, 0, 1)
    return np.dstack([np.interp(t, _DEEP[:, 0], _DEEP[:, c]) for c in (1, 2, 3)])       # RGB 0..255


def render_view(st: dict, view: str, raw: bool = False):
    """One zoomable picture of an alert cutout, as uint8 BGR.

    change      tonight in warm, before in cool: new light is orange, vanished light is blue, unchanged sky is white
    science     tonight's image          template   the reference from before
    difference  tonight minus before: brighter = gold, fainter = blue, no change = black
    Each panel is stretched to its own noise level (the survey's panels are not on one flux scale).
    """
    if view == "difference":
        d = np.clip(_up(_signed01(st["difference"]).astype(np.float32), raw), -1, 1)
        p, n = np.clip(d, 0, 1), np.clip(-d, 0, 1)
        rgb = np.dstack([255 * (p ** 0.8) + 40 * n ** 2, 200 * p ** 1.4 + 150 * n ** 1.2, 90 * p ** 3 + 255 * n ** 0.8])
    elif view == "change":
        s, t = (np.clip(_up(_asinh01(st[k]).astype(np.float32), raw), 0, 1) for k in ("science", "template"))
        rgb = np.dstack([s, (s + t) / 2, t]) * 255
    else:
        rgb = _deep(_up(_asinh01(st[view]).astype(np.float32), raw))
    return np.ascontiguousarray(np.clip(rgb, 0, 255).astype(np.uint8)[..., ::-1])


def describe(st: dict) -> dict:
    """What the cutouts hold: size, noise, and whether the centre brightened or faded."""
    d = np.nan_to_num(st["difference"].astype(np.float32))
    h, w = d.shape
    med, sig = sky_stats(d)
    c = max(2, min(h, w) // 8)
    core = d[h // 2 - c:h // 2 + c + 1, w // 2 - c:w // 2 + c + 1] - med
    peak = float(core.flat[np.abs(core).argmax()])
    return {"shape": [int(h), int(w)], "diff_snr": round(diff_snr(st["difference"]), 1), "peak_sigma": round(peak / sig, 1),
            "direction": "brightened" if peak > 0 else "faded", "version": HQ_VERSION}
