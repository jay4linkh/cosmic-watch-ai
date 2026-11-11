"""The vetting rules that keep stars, old AGN and asteroids out of the anomaly candidates."""
import time

from cosmic_watch import vetting

NOW = time.gmtime().tm_year
YY = f"{NOW % 100:02d}"


def test_first_seen_year_reads_ztf_names_only():
    assert vetting.first_seen_year("ztf", "ZTF18acwwehh") == 2018
    assert vetting.first_seen_year("lsst", "313994139530690582") is None
    assert vetting.first_seen_year("ztf", "not-a-ztf-name") is None


def test_tde_rejects_long_lived_and_archive_contradictions():
    assert not vetting.tde_ok("ztf", "ZTF18acwwehh")[0]                         # eight years old
    assert vetting.tde_ok("ztf", f"ZTF{YY}accxoia")[0]                          # this year
    assert not vetting.tde_ok("ztf", f"ZTF{YY}accxoia", "variable_star")[0]     # archives: a star
    assert vetting.tde_ok("lsst", "313695088848404576", "nuclear_transient")[0]


def test_hostless_needs_repeat_detection_and_asteroid_check():
    ok = dict(survey="lsst", oid="171145650184388632", n_det=3, broker_cls="SN", moving_ruled_out=True)
    assert vetting.hostless_ok(**ok)[0]
    assert not vetting.hostless_ok(**{**ok, "broker_cls": "asteroid"})[0]
    assert not vetting.hostless_ok(**{**ok, "n_det": 1})[0]
    assert not vetting.hostless_ok(**{**ok, "moving_ruled_out": False})[0]
    assert not vetting.hostless_ok(**{**ok, "survey": "ztf", "oid": "ZTF18acrxwjw"})[0]


def test_supernova_rejects_agn_broker_classes():
    assert vetting.supernova_ok("SN")[0]
    for cls in ("AGN", "QSO", "Blazar"):
        assert not vetting.supernova_ok(cls)[0]


def test_deep_population_drops_stars_and_cnn_rejections():
    dets = [{"label": "star", "cnn_label": None}, {"label": "galaxy", "cnn_label": "Not a galaxy"},
            {"label": "galaxy", "cnn_label": "Merging"}, {"label": "large galaxy", "cnn_label": None}]
    assert [d["cnn_label"] for d in vetting.deep_population(dets)] == ["Merging", None]


def test_moving_ruled_out_only_after_a_successful_check():
    ok = {"evidence": [{"source": "jpl_sbident", "claim_type": "catalog_fact",
                        "statement": "No known asteroid or comet within 10\" at the alert time (JPL ephemerides)."}]}
    failed = {"evidence": [{"source": "jpl_sbident", "claim_type": "missing", "statement": "JPL asteroid check failed (timeout)"}]}
    assert vetting.moving_ruled_out_from_report(ok)
    assert not vetting.moving_ruled_out_from_report(failed)
    assert not vetting.moving_ruled_out_from_report({"evidence": []})
