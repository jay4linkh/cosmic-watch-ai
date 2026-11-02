"""Offline tests for the restoration engine (synthetic images, no downloads)."""
import numpy as np
import pytest

from cosmic_watch.imaging import background, stretching, validation


def _field(seed=0, n=2000, size=512):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    sky = 5.0 + 0.004 * xx + 0.002 * yy                         # a gradient the background model must remove
    img = sky + rng.normal(0, 1.0, (size, size))
    for x, y in rng.uniform(10, size - 10, (60, 2)):            # stars
        img += 40 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 1.1 ** 2))
    gal = 3.0 * np.exp(-np.hypot(xx - 300, yy - 200) / 25)      # a big faint galaxy
    return (img + gal).astype(np.float32), sky, gal


def test_background_removes_gradient_keeps_galaxy():
    img, sky, gal = _field()
    valid = np.ones_like(img, bool)
    bg = background.estimate(img, valid)
    lin = img - bg["back"]
    assert abs(np.median((bg["back"] - sky)[~bg["source_mask"]])) < 0.15            # sky level recovered
    assert background.uniformity(lin, valid, bg["source_mask"]) < 0.2                   # flat afterwards
    r = np.hypot(*np.mgrid[0:512, 0:512][::-1] - np.array([300, 200])[:, None, None])
    kept = lin[r < 50].sum() / gal[r < 50].sum()
    assert kept > 0.8                                                                   # halo not eaten


def test_injection_recovery_passes_on_clean_field():
    img, _, _ = _field(seed=1)
    valid = np.ones_like(img, bool)
    bg = background.estimate(img, valid)
    lin = (img - bg["back"]).astype(np.float32)
    est = lambda im, v: background.estimate(im, v, mesh=bg["mesh"], thresh=bg["thresholds"][0],  # noqa: E731
                                            coarse_thresh=bg["thresholds"][1], adapt=False)
    res = validation.injection_test(img, valid, lin, bg["source_mask"], bg["global_rms"], est, n_point=30, n_disk=4)
    assert abs(res["point_flux_ratio_median"] - 1) < 0.03


def test_masked_pixels_never_filled():
    img, _, _ = _field(seed=2)
    valid = np.ones_like(img, bool)
    valid[:, :100] = False
    out = stretching.stretch(img - 5, valid, 1.0)
    assert out.dtype == np.uint8 and (out[:, :100] == 0).all()


@pytest.mark.parametrize("method", stretching.METHODS)
def test_stretch_methods_monotonic(method):
    x = np.linspace(-3, 200, 4096, dtype=np.float32).reshape(64, 64)
    out = stretching.stretch(x, np.ones_like(x, bool), 1.0, method=method)
    flat = out.ravel().astype(int)
    if method != "histeq":
        assert (np.diff(flat) >= 0).all()                     # brighter input never gets darker
    assert flat.max() > 200 and flat.min() < 30
