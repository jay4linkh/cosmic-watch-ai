"""Multi-scale science detector: finds tiny, medium and large diffuse galaxies, stays quiet on pure noise
(also when the noise is correlated, as in drizzled mosaics), and measures the flux that was put in."""
import cv2
import numpy as np
import pytest

from cosmic_watch.imaging import catalog_db, detect

R_PSF = 1.3


def disk(shape, x, y, r50, flux, q=1.0):
    yy, xx = np.mgrid[:shape[0], :shape[1]].astype(np.float32)
    h = np.hypot(r50, R_PSF) / 1.678
    p = np.exp(-np.hypot(xx - x, (yy - y) / q) / h)
    return p * (flux / p.sum())


def noise(shape, seed, correlated=False):
    n = np.random.default_rng(seed).normal(0, 1, shape).astype(np.float32)
    if correlated:
        n = cv2.GaussianBlur(n, (0, 0), 0.8)
        n /= n.std()
    return n


def scene(correlated=False):
    shape = (600, 600)
    img = noise(shape, 1, correlated)
    # correlated noise is truly noisier at the scale of a tiny source, so the same faint galaxy is rightly lost there
    truth = {"tiny": (120, 130, 1.5 * R_PSF, 450.0 if correlated else 150.0), "medium": (420, 160, 6 * R_PSF, 1500.0), "large": (300, 420, 22 * R_PSF, 9000.0)}
    for x, y, r, f in truth.values():
        img += disk(shape, x, y, r, f)
    return np.ascontiguousarray(img), np.ones(shape, np.float32), np.ones(shape, bool), truth


def nearest(cat, x, y):
    d = np.hypot(cat["x"] - x, cat["y"] - y)
    return int(d.argmin()), float(d.min())


@pytest.mark.parametrize("correlated", [False, True])
def test_finds_every_size(correlated):
    lin, rms, valid, truth = scene(correlated)
    cat, rep, _ = detect.run_passes(lin, rms, valid, valid, 2 * R_PSF)
    for name, (x, y, r, _) in truth.items():
        assert nearest(cat, x, y)[1] < max(3, 0.5 * r), f"{name} galaxy not found"
    assert len(cat["x"]) <= 6, "noise peaks reported as sources"


@pytest.mark.parametrize("correlated", [False, True])
def test_quiet_on_pure_noise(correlated):
    shape = (700, 700)
    lin = np.ascontiguousarray(noise(shape, 5, correlated))
    cat, _, _ = detect.run_passes(lin, np.ones(shape, np.float32), np.ones(shape, bool), np.ones(shape, bool), 2 * R_PSF)
    assert len(cat["x"]) <= 3


def test_flux_and_size_class():
    lin, rms, valid, truth = scene()
    cat, _, _ = detect.run_passes(lin, rms, valid, valid, 2 * R_PSF)
    cat = detect.measure(cat, lin, rms, valid, R_PSF, 25.0, {})
    i, _ = nearest(cat, *truth["medium"][:2])
    assert abs(cat["flux"][i] / truth["medium"][3] - 1) < 0.15            # Kron aperture: most of an exponential disk
    assert abs(cat["mag"][i] - (25.0 - 2.5 * np.log10(cat["flux"][i]))) < 1e-6
    assert cat["size_class"][i] in ("small", "medium") and "galaxy" in cat["label"][i]
    j, _ = nearest(cat, *truth["large"][:2])
    assert cat["size_class"][j] in ("large", "giant")
    assert cat["r50_px"][j] > cat["r50_px"][i]


def test_masked_pixels_give_no_sources():
    lin, rms, valid, truth = scene()
    valid = valid.copy()
    valid[:, :250] = False                                              # hides the tiny galaxy at x = 120
    cat, _, _ = detect.run_passes(np.where(valid, lin, 0).astype(np.float32), rms, valid, valid, 2 * R_PSF)
    assert not (cat["x"] < 245).any()


def test_frame_flip():
    cat = {"x": np.array([10.0]), "y": np.array([0.0]), "a": np.array([4.0]), "b": np.array([2.0]), "theta": np.array([0.3])}
    x, y, a, b, th = catalog_db._to_frame(cat, (200, 400), (200, 100), flip=True)
    assert abs(x[0] - 4.75) < 1e-6 and abs(y[0] - 99.25) < 1e-6 and a[0] == 2.0 and th[0] == -0.3
    assert catalog_db._to_frame(cat, (200, 400), (200, 100), flip=False)[1][0] == pytest.approx(-0.25)
