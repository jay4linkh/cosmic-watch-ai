"""Sunspot and active-region detection on SDO full-disk images (OpenCV)."""
import cv2
import numpy as np


def _disk(gray):
    """Locate the solar disk: centre and radius from the largest bright blob."""
    _, th = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        h, w = gray.shape
        return (w / 2, h / 2), min(h, w) * 0.4
    (cx, cy), r = cv2.minEnclosingCircle(max(contours, key=cv2.contourArea))
    return (cx, cy), r


def _circle_mask(shape, centre, radius):
    mask = np.zeros(shape, np.uint8)
    cv2.circle(mask, (int(centre[0]), int(centre[1])), int(radius), 1, -1)
    return mask.astype(bool)


def _regions(binary, min_area, intensity, label):
    n, lab, st, cen = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    out = []
    for i in range(1, n):
        area = int(st[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x, y, w, h = (int(st[i, k]) for k in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
        out.append({
            "label": label, "x": float(cen[i][0]), "y": float(cen[i][1]),
            "a": w / 2, "b": h / 2, "theta": 0.0,
            "score": float(intensity[lab == i].mean()), "area": area,
        })
    return sorted(out, key=lambda d: d["area"], reverse=True)


def sunspots(path):
    """Dark spots on the visible-light (HMI continuum) disk."""
    bgr = cv2.imread(str(path))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    centre, r = _disk(gray)
    disk = _circle_mask(gray.shape, centre, r * 0.96)    # skip the dark limb
    # Divide out limb darkening with a large median filter, then look for dark dips.
    smooth = cv2.medianBlur(gray, 61).astype(np.float32) + 1
    ratio = gray.astype(np.float32) / smooth
    spots = (ratio < 0.82) & disk
    spots = cv2.morphologyEx(spots.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    regs = _regions(spots, min_area=4, intensity=1 - ratio, label="sunspot")
    stats = {"n_sunspots": len(regs), "spot_area_px": int(sum(d["area"] for d in regs)),
             "disk_radius_px": round(r, 1)}
    return bgr, regs, stats


def active_regions(path, n_sigma: float = 3.0):
    """Bright magnetically active regions in the extreme-UV corona (AIA).

    The chromosphere (304 Å) is mottled all over, so call it with a higher n_sigma.
    """
    bgr = cv2.imread(str(path))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    centre, r = _disk(gray)
    disk = _circle_mask(gray.shape, centre, r * 0.98)
    vals = gray[disk].astype(np.float32)
    med = float(np.median(vals))
    sigma = 1.4826 * float(np.median(np.abs(vals - med))) + 1e-3
    thr = med + n_sigma * sigma
    bright = (gray > thr) & disk
    bright = cv2.morphologyEx(bright.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    bright = cv2.dilate(bright, np.ones((9, 9), np.uint8))       # merge loop systems into one region
    regs = _regions(bright & disk, min_area=150, intensity=gray.astype(np.float32), label="active region")
    stats = {
        "n_active_regions": len(regs),
        "activity_index": round(100.0 * float(((gray > thr) & disk).sum()) / float(disk.sum()), 2),
        "peak_brightness": int(gray[disk].max()),
        "disk_radius_px": round(r, 1),
    }
    return bgr, regs, stats


def annotate(bgr, regs, out_path):
    img = bgr.copy()
    for i, d in enumerate(regs[:40], 1):
        color = (0, 255, 255) if d["label"] == "sunspot" else (0, 255, 0)
        pad = 6
        x0, y0 = int(d["x"] - d["a"] - pad), int(d["y"] - d["b"] - pad)
        x1, y1 = int(d["x"] + d["a"] + pad), int(d["y"] + d["b"] + pad)
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2, cv2.LINE_AA)
        cv2.putText(img, str(i), (x0, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return out_path
