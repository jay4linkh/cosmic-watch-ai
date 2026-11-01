"""Offline tests for the cross-survey engine (no network: catalogues are stubbed)."""
import io
import math

import numpy as np
import pytest

from cosmic_watch.xsurvey import catalogs, coverage, engine, photometry, wcs
from cosmic_watch.xsurvey.match import best, match, propagate, sep_arcsec


# ---------- matching ----------

def test_sep_small_angle():
    assert sep_arcsec(10, 0, 10 + 1 / 3600, 0) == pytest.approx(1.0, rel=1e-6)
    assert sep_arcsec(10, 60, 10 + 2 / 3600, 60) == pytest.approx(1.0, rel=1e-4)     # cos(dec) shrink
    assert sep_arcsec(359.9999, 0, 0.0001, 0) == pytest.approx(0.72, rel=1e-3)          # RA wrap


def test_propagate_proper_motion():
    ra, dec = propagate(100.0, 0.0, 1000.0, -500.0, 2016.0, 2026.0)                    # 1"/yr east, 0.5"/yr south
    assert (ra - 100.0) * 3600 == pytest.approx(10.0, rel=1e-6)
    assert dec * 3600 == pytest.approx(-5.0, rel=1e-6)
    assert propagate(1, 2, None, None, 2016, 2026) == (1, 2)


def test_match_prefers_close_and_flags_blends():
    t = {"ra": 150.0, "dec": 2.0, "err": 0.2}
    near = {"id": "a", "ra": 150.0 + 0.1 / 3600, "dec": 2.0, "err": 0.1}
    far = {"id": "b", "ra": 150.0 + 4 / 3600, "dec": 2.0, "err": 0.1}
    ms = match(t, [far, near], field_count=20, field_radius_arcsec=60)
    assert ms[0]["id"] == "a" and ms[0]["p_match"] > 0.95 and ms[1]["p_match"] < 0.01
    assert not ms[0]["ambiguous"]
    twin = {"id": "c", "ra": 150.0 - 0.1 / 3600, "dec": 2.0, "err": 0.1}
    assert match(t, [near, twin], field_count=20, field_radius_arcsec=60)[0]["ambiguous"]


def test_match_density_matters():
    t = {"ra": 0.0, "dec": 0.0, "err": 0.5}
    c = [{"id": "x", "ra": 1.0 / 3600, "dec": 0.0, "err": 0.5}]
    sparse = match(t, c, field_count=2, field_radius_arcsec=60)[0]["p_match"]
    crowded = match(t, c, field_count=3000, field_radius_arcsec=60)[0]["p_match"]
    assert sparse > crowded


def test_match_applies_proper_motion():
    t = {"ra": 100.0, "dec": 0.0, "err": 0.1, "epoch": 2026.0}
    star = {"id": "s", "ra": 100.0 - 10 / 3600, "dec": 0.0, "err": 0.05, "epoch": 2016.0, "pmra": 1000.0, "pmdec": 0.0,
            "pmra_err": 0.05, "pmdec_err": 0.05}
    m = match(t, [star], field_count=10, field_radius_arcsec=60)[0]
    assert m["pm_corrected"] and m["sep_arcsec"] < 0.01 and m["p_match"] > 0.99
    assert best([m])["id"] == "s"


# ---------- photometry / coverage ----------

def test_photometry():
    m, e = photometry.njy_to_ab(3631e9 / 1e9 * 1e9, 1.0)        # 3631 Jy -> 0 mag AB (31.4 zero point in nJy)
    assert m == pytest.approx(31.4 - 2.5 * math.log10(3631e9), abs=1e-6)
    r, sys = photometry.gaia_g_to_r(15.0, None)
    assert sys == photometry.SYS["gaia_nocolor"] and r == 15.1
    d, sig, s = photometry.compare(18.0, 0.02, 0.05, 17.0, 0.02, 0.05)
    assert d == -1.0 and sig
    assert not photometry.compare(18.0, 0.02, 0.3, 17.8, 0.02, 0.05)[1]        # within the Gaia conversion systematic
    assert photometry.flux_ratio(20.0, 20.0) == pytest.approx(1.0)
    assert photometry.flux_ratio(22.5, 20.0) == pytest.approx(0.1, rel=1e-3)


def test_coverage():
    assert coverage.status("ps1_dr2", -45, 100) == "not_covered"
    assert coverage.status("ps1_dr2", 10, 0) == "not_covered"
    assert coverage.status("ps1_dr2", 10, None) == "uncertain"
    assert coverage.status("ls_dr10", 10, 5) == "covered"
    assert coverage.should_have_seen("ls_dr10", "r", 21.0)
    assert not coverage.should_have_seen("ps1_dr2", "r", 21.6)


# ---------- WCS ----------

def _fake_fits(nx=200, ny=100, scale=0.1):
    from astropy.io import fits
    prim = fits.PrimaryHDU()
    sci = fits.ImageHDU(np.zeros((ny, nx), dtype=np.float32), name="SCI")
    h = sci.header
    h.update(CTYPE1="RA---TAN", CTYPE2="DEC--TAN", CRVAL1=150.0, CRVAL2=2.0, CRPIX1=1.0, CRPIX2=1.0,
             CD1_1=-scale / 3600, CD1_2=0.0, CD2_1=0.0, CD2_2=scale / 3600)
    buf = io.BytesIO()
    fits.HDUList([prim, sci]).writeto(buf)
    return buf.getvalue()


def test_wcs_headers_and_preview_mapping():
    hdrs = wcs.parse_headers(_fake_fits()[:20000])          # a byte prefix is enough
    sci = next(h for h in hdrs if h.get("EXTNAME") == "SCI")
    # preview at half size, y down: the bottom-left preview corner is FITS pixel (1, 1) = CRVAL
    ra, dec = wcs.preview_to_sky(sci, (100, 50), [(0.25, 49.75)])
    assert ra[0] == pytest.approx(150.0, abs=1e-6) and dec[0] == pytest.approx(2.0, abs=1e-6)
    ra2, dec2 = wcs.preview_to_sky(sci, (100, 50), [(0.25, 39.75)])      # 10 preview px up = 20 sci px = 2"
    assert (dec2[0] - dec[0]) * 3600 == pytest.approx(2.0, rel=1e-3)
    assert wcs.pixel_scale_arcsec(sci, (100, 50)) == pytest.approx(0.2, rel=1e-3)


# ---------- engine verdicts (stubbed catalogues) ----------

def _cat(sources, n=None):
    return {"ok": True, "sources": sources, "field_count": len(sources) if n is None else n, "provenance": {}}


def _src(ra, dec, **kw):
    s = {"id": kw.pop("id", f"{ra:.6f}"), "ra": ra, "dec": dec, "err": 0.05, "mags": {}, "morph": None}
    s.update(kw)
    return s


def _stub(monkeypatch, cats):
    full = {n: _cat([]) for n in catalogs.CATALOGS}
    full.update(cats)
    monkeypatch.setattr(catalogs, "query_all", lambda *a, **k: full)
    monkeypatch.setattr(catalogs, "jpl_sbident", lambda *a, **k: _cat([]))


T = {"kind": "alert", "ref": "alert:test:1", "ra": 150.0, "dec": 2.0, "err": 0.2, "epoch": 2026.8, "mjd": 61340.0, "n_det": 5,
     "mag_now": 19.0, "survey": "ztf", "oid": "ZTFtest"}


def test_verdict_star(monkeypatch):
    star = _src(150.0, 2.0, parallax=5.0, parallax_err=0.1, pmra=1.0, pmdec=1.0, pmra_err=0.1, pmdec_err=0.1, epoch=2016.0,
                mags={"G": [17.0, 0.01]}, morph="point", extra={})
    _stub(monkeypatch, {"gaia_dr3": _cat([star], 30), "ls_dr10": _cat([_src(150.0, 2.0, morph="point", extra={"type": "PSF"},
                                                                              mags={"r": [17.2, 0.01]})], 30)})
    rep = engine.analyze(dict(T), persist=False)
    assert rep["verdict"] == "variable_star"
    assert any("parallax" in e["statement"] for e in rep["evidence"])


def test_verdict_supernova_with_host_and_absolute_mag(monkeypatch):
    gal = _src(150.0 + 2 / 3600, 2.0, morph="extended", redshift=0.05, redshift_type="spec",
               extra={"type": "EXP", "shape_r": 2.0}, mags={"r": [17.5, 0.01]})
    _stub(monkeypatch, {"ls_dr10": _cat([gal], 40)})
    rep = engine.analyze(dict(T), persist=False)
    assert rep["verdict"] == "supernova_candidate"
    assert rep["host"]["d_dlr"] == pytest.approx(1.0, abs=0.05)
    assert "absolute magnitude" in rep["summary"]


def test_verdict_hostless_needs_depth(monkeypatch):
    field = [_src(150.0 + 30 / 3600, 2.0, morph="point", extra={})]                  # survey covered, nothing at target
    _stub(monkeypatch, {"ls_dr10": _cat(field, 50), "ps1_dr2": _cat(field, 50)})
    assert engine.analyze(dict(T), persist=False)["verdict"] == "new_hostless_transient"
    faint = dict(T, mag_now=23.9)                                                      # fainter than every catalogue
    assert engine.analyze(faint, persist=False)["verdict"] == "new_transient_unconstrained"


def test_failed_service_is_missing_not_negative(monkeypatch):
    _stub(monkeypatch, {"ls_dr10": {"ok": False, "error": "timeout", "sources": [], "field_count": None}})
    rep = engine.analyze(dict(T), persist=False)
    assert rep["coverage"]["ls_dr10"] == "failed"
    assert any(e["claim_type"] == "missing" and "Legacy" in e["statement"] for e in rep["evidence"])


def test_asteroid_wins(monkeypatch):
    _stub(monkeypatch, {})
    monkeypatch.setattr(catalogs, "jpl_sbident", lambda *a, **k: _cat([{"id": "(1) Ceres", "name": "(1) Ceres", "sep": 1.5, "V": 8.0,
                                                                         "rate_ra": 30.0, "rate_dec": 0.0}]))
    rep = engine.analyze(dict(T, n_det=1), persist=False)
    assert rep["verdict"] == "known_asteroid" and rep["confidence"] > 0.9


def test_detection_disagreement_is_reported(monkeypatch):
    gal = _src(150.0, 2.0, morph="extended", extra={"type": "SER", "shape_r": 1.0})
    _stub(monkeypatch, {"ls_dr10": _cat([gal], 50)})
    t = {"kind": "detection", "ref": "detection:1:1", "ra": 150.0, "dec": 2.0, "err": 0.3, "label": "star", "cnn_label": None}
    rep = engine.analyze(t, persist=False)
    assert rep["verdict"] == "catalog_galaxy" and rep["cnn_agrees"] is False


def test_persist_roundtrip(monkeypatch, tmp_path):
    import sqlite3

    from cosmic_watch import db
    con = sqlite3.connect(tmp_path / "t.db")
    con.row_factory = sqlite3.Row
    con.executescript(db.SCHEMA)
    _stub(monkeypatch, {"ls_dr10": _cat([_src(150.0, 2.0, morph="point", extra={"type": "PSF"})], 10)})
    rep = engine.analyze(dict(T), con=con)
    assert engine.load(con, T["ref"])["verdict"] == rep["verdict"]
    assert con.execute("SELECT COUNT(*) FROM scientific_evidence").fetchone()[0] == len(rep["evidence"])
    engine.analyze(dict(T), con=con)                                                     # re-run replaces, never duplicates
    assert con.execute("SELECT COUNT(*) FROM astronomical_objects").fetchone()[0] == 1
    assert con.execute("SELECT COUNT(*) FROM scientific_evidence").fetchone()[0] == len(rep["evidence"])
