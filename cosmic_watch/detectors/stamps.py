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
