"""Did a survey actually look here, and how faint could it see? (absence of evidence vs evidence of absence)

A non-detection only means something if (a) the survey covered this position and (b) it could have seen an
object as bright as the one we now see. Footprints and depths below are approximate published values; coverage
is also checked empirically: if a catalogue returns no sources at all in a 1-arcminute cone, we treat the
position as not covered (or unusable) rather than as "empty sky".
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Survey:
    name: str
    label: str
    dec_min: float
    dec_max: float
    depth: dict            # band -> approximate 5-sigma limiting magnitude of the catalogue
    epoch: tuple           # (start, end) decimal years of the imaging
    resolution_arcsec: float


SURVEYS = {
    "gaia_dr3": Survey("gaia_dr3", "Gaia DR3", -90, 90, {"G": 20.7}, (2014.6, 2017.4), 0.2),
    "ps1_dr2": Survey("ps1_dr2", "Pan-STARRS1 DR2 (mean)", -30, 90, {"g": 22.0, "r": 21.8, "i": 21.5}, (2010.4, 2014.2), 1.1),
    "sdss_dr18": Survey("sdss_dr18", "SDSS DR18", -20, 85, {"r": 22.2, "g": 22.2}, (2000.0, 2009.0), 1.4),
    "ls_dr10": Survey("ls_dr10", "DESI Legacy Surveys DR10", -90, 85, {"g": 24.0, "r": 23.4, "z": 22.5}, (2014.0, 2021.0), 1.2),
    "alerce_ztf": Survey("alerce_ztf", "ZTF (ALeRCE)", -31, 90, {"r": 20.5, "g": 20.8}, (2018.2, 2026.9), 2.0),
}


def footprint(survey, dec):
    s = SURVEYS.get(survey)
    return bool(s and s.dec_min <= dec <= s.dec_max)


def status(survey, dec, field_count):
    """covered | not_covered | uncertain."""
    if not footprint(survey, dec):
        return "not_covered"
    if field_count is None:
        return "uncertain"
    return "covered" if field_count > 0 else "not_covered"


def should_have_seen(survey, band, mag_now, margin=0.5):
    """Could this survey have detected a source as bright as mag_now (with some margin)?"""
    s = SURVEYS.get(survey)
    if not s or mag_now is None:
        return False
    depth = s.depth.get(band) or s.depth.get("r") or next(iter(s.depth.values()))
    return mag_now < depth - margin
