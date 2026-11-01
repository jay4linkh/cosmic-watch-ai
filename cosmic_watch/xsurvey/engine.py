"""Cross-survey intelligence: everything the archives know about one sky position, as a cited evidence report.

    target (alert | detection | gcn | position)
      -> catalogue cones (Gaia, Pan-STARRS, SDSS, Legacy Surveys, SIMBAD, NED, ZTF)       catalogs.py
      -> probabilistic counterpart per catalogue, proper motion applied, blends flagged      match.py
      -> unified history: archival epochs, upper limits where a survey looked and saw nothing coverage.py / photometry.py
      -> deterministic verdict with confidence, every claim typed and sourced
      -> persisted (astronomical_objects, cross_survey_matches, historical_observations, scientific_evidence)

Claim types: catalog_fact (a catalogue says so), measurement (our own numbers), model_prediction (a classifier's
output), inference (a rule combining facts; always shows its reasons), missing (a source was down or did not cover
the position). Nothing is invented: when a service fails, the report says so and the verdict loses confidence.
"""
import json
import logging
import math
import time

from . import catalogs, coverage, photometry
from .match import best, match, sep_arcsec

log = logging.getLogger("cosmic_watch")
ENGINE_VERSION = "xsurvey-1.0"
CAND_RADIUS = 10.0                    # arcsec: counterpart candidates (beyond this nothing is "the same source")
POS_CATALOGS = ["gaia_dr3", "ps1_dr2", "sdss_dr18", "ls_dr10"]
NAME_CATALOGS = ["simbad", "ned"]
AGN_TYPES = ("QSO", "AGN", "Sy1", "Sy2", "SyG", "Bla", "BLL", "LIN", "Q?", "AG?", "Bz?", "BL?")
GALAXY_LABELS = {"galaxy", "large galaxy", "Spiral", "Elliptical", "Edge-on disk", "Merger", "Irregular",
                 "Lens/arc", "Smooth round", "Smooth cigar", "Featured", "Disturbed"}


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def mjd_to_year(mjd):
    return 2000.0 + (mjd - 51544.5) / 365.25


class Report:
    def __init__(self, target):
        self.target = target
        self.evidence, self.history, self.matches, self.flags, self.asteroids = [], [], {}, [], []

    def claim(self, claim_type, statement, source=None, confidence=None, **data):
        self.evidence.append({"claim_type": claim_type, "statement": statement, "source": source,
                              "confidence": None if confidence is None else round(confidence, 3), "data": data or None})


# ---------- per-catalogue counterparts ----------

def _counterparts(rep, cats):
    t = rep.target
    radius = max(CAND_RADIUS, 5 * (t.get("err") or 0.3))
    for name in POS_CATALOGS + NAME_CATALOGS:
        res = cats.get(name)
        label = catalogs.LABELS[name]
        if not res or not res["ok"]:
            rep.claim("missing", f"{label} could not be queried ({(res or {}).get('error', 'not run')}); "
                                 "its information is absent, not negative.", name)
            rep.matches[name] = {"status": "failed", "matches": []}
            continue
        box = (radius + 30) / 3600
        cosd = max(math.cos(math.radians(t["dec"])), 1e-3)
        cand = [s for s in res["sources"] if abs(s["dec"] - t["dec"]) < box and abs((s["ra"] - t["ra"] + 180) % 360 - 180) * cosd < box
                and sep_arcsec(t["ra"], t["dec"], s["ra"], s["dec"]) <= radius + 30 * bool(s.get("pmra"))]
        n = res["field_count"] if res["field_count"] is not None else len(res["sources"])
        fr = res.get("provenance", {}).get("radius_arcsec") or res.get("field_radius_arcsec") or catalogs.FIELD_RADIUS
        ms = [m for m in match(t, cand, field_count=n, field_radius_arcsec=fr) if m["sep_arcsec"] <= radius]
        cov = coverage.status(name, t["dec"], n) if name in coverage.SURVEYS else ("covered" if name in NAME_CATALOGS else "uncertain")
        rep.matches[name] = {"status": cov, "field_count": n, "matches": ms[:5], "best": best(ms),
                             "provenance": res.get("provenance"), "fetched_at": res.get("fetched_at")}


def _identity(rep):
    """What kind of thing sits at the position, according to each catalogue."""
    m = rep.matches
    ident = {}
    g = (m.get("gaia_dr3") or {}).get("best")
    if g:
        plx, eplx = g.get("parallax"), g.get("parallax_err")
        pm = math.hypot(g.get("pmra") or 0, g.get("pmdec") or 0)
        epm = math.hypot(g.get("pmra_err") or 0, g.get("pmdec_err") or 0) or None
        ident["gaia"] = {"p": g["p_match"], "sep": g["sep_arcsec"], "G": (g["mags"].get("G") or [None])[0],
                         "plx_snr": plx / eplx if plx is not None and eplx else None, "parallax": plx,
                         "pm": pm if g.get("pmra") is not None else None, "pm_snr": pm / epm if epm else None,
                         "p_quasar": (g.get("extra") or {}).get("p_quasar"), "p_galaxy": (g.get("extra") or {}).get("p_galaxy")}
        txt = f"Gaia DR3 {g['id']} lies {g['sep_arcsec']:.2f}\" away (P(same source) = {g['p_match']:.2f}"
        txt += ", position moved to the target epoch with its proper motion)" if g["pm_corrected"] and (g.get("pmra_err") or 9) < 2             and math.hypot(g["pmra"], g["pmdec"]) > 3 * math.hypot(g["pmra_err"], g["pmdec_err"]) else ")"
        if ident["gaia"]["plx_snr"] and ident["gaia"]["plx_snr"] > 5:
            txt += f"; parallax {plx:.2f} ± {eplx:.2f} mas, i.e. about {1 / max(plx, 1e-3):.2f} kpc away inside the Milky Way"
        if ident["gaia"]["pm_snr"] and ident["gaia"]["pm_snr"] > 5:
            txt += f"; proper motion {pm:.1f} mas/yr"
        rep.claim("catalog_fact", txt + ".", "gaia_dr3", g["p_match"], source_id=g["id"])
    ls = (m.get("ls_dr10") or {}).get("best")
    if ls:
        ex = ls.get("extra") or {}
        ident["ls"] = {"p": ls["p_match"], "sep": ls["sep_arcsec"], "morph": ls["morph"], "type": ex.get("type"),
                       "r": (ls["mags"].get("r") or [None])[0], "z": ls.get("redshift"), "z_type": ls.get("redshift_type"),
                       "z_err": ex.get("z_phot_std"), "shape_r": ex.get("shape_r")}
        z = f", {ls['redshift_type']} redshift {ls['redshift']:.3f}" if ls.get("redshift") is not None else ""
        rep.claim("catalog_fact", f"Legacy Surveys DR10 source {ls['id']} ({ex.get('type')} model, "
                                  f"{'point-like' if ls['morph'] == 'point' else 'resolved'}) at {ls['sep_arcsec']:.2f}\"{z}.",
                  "ls_dr10", ls["p_match"], ls_id=ls["id"])
    for name, key in (("ps1_dr2", "ps1"), ("sdss_dr18", "sdss")):
        b = (m.get(name) or {}).get("best")
        if b:
            ident[key] = {"p": b["p_match"], "sep": b["sep_arcsec"], "morph": b["morph"], "r": (b["mags"].get("r") or [None])[0],
                          "z": b.get("redshift"), "otype": b.get("otype")}
            spec = f", SDSS spectrum: {b['otype']} at z = {b['redshift']:.4f}" if b.get("redshift") is not None and b.get("otype") else ""
            rep.claim("catalog_fact", f"{catalogs.LABELS[name]} {b['id']} at {b['sep_arcsec']:.2f}\" "
                                      f"({b['morph'] or 'unknown'} morphology){spec}.", name, b["p_match"], cat_id=b["id"])
    for name in NAME_CATALOGS:
        b = (m.get(name) or {}).get("best") or next((c for c in (m.get(name) or {}).get("matches", []) if c["sep_arcsec"] < 2), None)
        if b:
            ident[name] = {"name": b["name"], "otype": b.get("otype"), "z": b.get("redshift"), "sep": b["sep_arcsec"]}
            z = f", redshift {b['redshift']:.4f}" if b.get("redshift") is not None else ""
            rep.claim("catalog_fact", f"{catalogs.LABELS[name]}: {b['name']} (type {b.get('otype')}) at {b['sep_arcsec']:.2f}\"{z}.",
                      name, b.get("p_match"))
    for name in POS_CATALOGS:
        ms = (rep.matches.get(name) or {}).get("matches") or []
        if ms and ms[0]["ambiguous"]:
            rep.flags.append(f"blend:{name}")
            rep.claim("inference", f"{catalogs.LABELS[name]} has {sum(1 for x in ms if x['p_match'] >= 0.2)} sources that could each be "
                                   "the counterpart: the position is blended, so the match is uncertain.", name)
    return ident


def _host(rep, cats):
    """Most likely host galaxy: smallest separation in units of the galaxy's half-light radius (DLR method)."""
    res = cats.get("ls_dr10")
    if not res or not res["ok"]:
        return None
    t, best_h = rep.target, None
    for s in res["sources"]:
        r = (s.get("extra") or {}).get("shape_r")
        if s["morph"] != "extended" or not r:
            continue
        sep = sep_arcsec(t["ra"], t["dec"], s["ra"], s["dec"])
        d = sep / max(r, 0.3)
        if sep < 30 and (best_h is None or d < best_h["d_dlr"]):
            best_h = {"id": s["id"], "sep": round(sep, 2), "d_dlr": round(d, 2), "shape_r": r, "z": s.get("redshift"),
                      "z_type": s.get("redshift_type"), "z_err": (s.get("extra") or {}).get("z_phot_std"),
                      "r": (s["mags"].get("r") or [None])[0], "ra": s["ra"], "dec": s["dec"]}
    if best_h and best_h["d_dlr"] <= 4:
        z = (f", {best_h['z_type']} redshift {best_h['z']:.3f}" + (f" ± {best_h['z_err']:.3f}" if best_h.get("z_err") else "")
             if best_h["z"] is not None else "")
        rep.claim("inference", f"Likely host galaxy: Legacy Surveys {best_h['id']}, {best_h['sep']}\" away = "
                               f"{best_h['d_dlr']} half-light radii{z}.", "ls_dr10", max(0.2, 1 - best_h["d_dlr"] / 5), **best_h)
        return best_h
    return None


# ---------- history ----------

def _archival_r(ident, rep):
    """Best archival r-band brightness (mag, err, systematic, survey, epoch, estimated)."""
    m = rep.matches
    for name, key in (("ps1_dr2", "ps1"), ("ls_dr10", "ls"), ("sdss_dr18", "sdss")):
        b = (m.get(name) or {}).get("best")
        if b and b["mags"].get("r") and b["mags"]["r"][0] is not None:
            return {"mag": b["mags"]["r"][0], "err": b["mags"]["r"][1], "sys": photometry.SYS["same"], "survey": name,
                    "epoch": b.get("epoch"), "estimated": False}
    g = (m.get("gaia_dr3") or {}).get("best")
    if g and g["mags"].get("G"):
        r, sys = photometry.gaia_g_to_r(g["mags"]["G"][0], g.get("bp_rp"))
        return {"mag": r, "err": 0.01, "sys": sys, "survey": "gaia_dr3", "epoch": g.get("epoch"), "estimated": True}
    return None


def _history(rep, cats, con):
    t = rep.target
    for name in POS_CATALOGS:
        info = rep.matches.get(name) or {}
        b, sv = info.get("best"), coverage.SURVEYS[name]
        if b:
            for band, (mag, err) in (b.get("mags") or {}).items():
                if mag is not None:
                    rep.history.append({"survey": name, "epoch": b.get("epoch"), "band": band, "mag": mag, "mag_err": err,
                                        "upper_limit": 0, "estimated": 0, "source": f"{catalogs.LABELS[name]} {b['id']}"})
        elif info.get("status") == "covered":
            band = "G" if name == "gaia_dr3" else "r"
            depth = sv.depth.get(band)
            rep.history.append({"survey": name, "epoch": sum(sv.epoch) / 2, "band": band, "mag": depth, "mag_err": None,
                                "upper_limit": 1, "estimated": 1,
                                "source": f"{sv.label}: no source within {CAND_RADIUS:.0f}\" (catalogue limit ~{depth})"})
    z = cats.get("alerce_ztf")
    if z and z["ok"]:
        for s in z["sources"]:
            if s["id"] == t.get("oid"):
                continue
            ex = s["extra"]
            rep.history.append({"survey": "alerce_ztf", "mjd": ex.get("firstmjd"), "epoch": s.get("epoch"), "band": None, "mag": None,
                                "mag_err": None, "upper_limit": 0, "estimated": 0,
                                "source": f"ZTF object {s['id']}: {ex.get('ndet')} detections, MJD {ex.get('firstmjd'):.0f}–{ex.get('lastmjd'):.0f}"})
            rep.claim("catalog_fact", f"ZTF already saw something here: {s['id']} with {ex.get('ndet')} detections "
                                      f"(first MJD {ex.get('firstmjd'):.0f}).", "alerce_ztf", None, oid=s["id"])
    if con is not None:
        d = 2 / 3600
        for a in con.execute("SELECT survey, oid, cls, lastmjd, ra, dec FROM alerts WHERE ra BETWEEN ? AND ? AND dec BETWEEN ? AND ?",
                             (t["ra"] - d / max(math.cos(math.radians(t["dec"])), 1e-3), t["ra"] + d / max(math.cos(math.radians(t["dec"])), 1e-3),
                              t["dec"] - d, t["dec"] + d)):
            if a["oid"] != t.get("oid") and sep_arcsec(t["ra"], t["dec"], a["ra"], a["dec"]) < 2:
                rep.history.append({"survey": a["survey"], "mjd": a["lastmjd"], "epoch": mjd_to_year(a["lastmjd"]) if a["lastmjd"] else None,
                                    "band": None, "mag": None, "mag_err": None, "upper_limit": 0, "estimated": 0,
                                    "source": f"our {a['survey'].upper()} alert {a['oid']} ({a['cls']})"})
    for p in (t.get("light_curve") or [])[:1] + (t.get("light_curve") or [])[-1:]:
        rep.history.append({"survey": t.get("survey"), "mjd": p["mjd"], "epoch": mjd_to_year(p["mjd"]), "band": p["band"], "mag": p["mag"],
                            "mag_err": p.get("err"), "upper_limit": 0, "estimated": 0, "source": f"alert light curve {t.get('oid')}"})
    rep.history.sort(key=lambda h: h.get("epoch") or 0)


# ---------- verdict ----------

def _abs_mag(m, z):
    if m is None or z is None or z <= 0.002:
        return None
    from astropy.cosmology import Planck18
    return m - Planck18.distmod(z).value + 2.5 * math.log10(1 + z)          # rough K-correction-free


def _verdict_transient(rep, ident, host):
    """Alerts / afterglows: what is this flux change most likely to be?"""
    t, cands = rep.target, []
    for b in rep.asteroids:
        cands.append(("known_asteroid", 0.97 if b["sep"] < 5 else 0.85,
                      f"A known Solar System body, {b['name']}, was predicted {b['sep']:.1f}\" from this position at the alert time "
                      f"(V ≈ {b['V']}, moving {math.hypot(b['rate_ra'] or 0, b['rate_dec'] or 0):.0f}\"/hour): this is an asteroid, not a "
                      "change in a fixed object."))
    g = ident.get("gaia") or {}
    star_ev = []
    if g.get("plx_snr") and g["plx_snr"] > 5:
        star_ev.append(f"Gaia parallax at {g['plx_snr']:.0f} sigma")
    if g.get("pm_snr") and g["pm_snr"] > 5 and (g.get("pm") or 0) > 5:
        star_ev.append(f"proper motion {g['pm']:.0f} mas/yr")
    if (ident.get("simbad") or {}).get("otype") and "*" in ident["simbad"]["otype"] and ident["simbad"]["sep"] < 1.5:
        star_ev.append(f"SIMBAD type {ident['simbad']['otype']}")
    if (ident.get("ls") or {}).get("type") == "PSF" and (ident["ls"].get("r") or 99) < 19 and ident["ls"]["p"] > 0.8:
        star_ev.append("bright point source in Legacy Surveys")
    agn_ev = []
    for k in ("simbad", "ned"):
        o = (ident.get(k) or {}).get("otype") or ""
        if any(a in o for a in AGN_TYPES) and ident[k]["sep"] < 1.5:
            agn_ev.append(f"{catalogs.LABELS[k]} type {o}")
    if (ident.get("sdss") or {}).get("otype") == "QSO":
        agn_ev.append("SDSS spectrum classified QSO")
    if (g.get("p_quasar") or 0) > 0.5:
        agn_ev.append(f"Gaia quasar probability {g['p_quasar']:.2f}")
    if star_ev:
        cands.append(("variable_star", min(0.97, 0.6 + 0.12 * len(star_ev)),
                      "Variable star in the Milky Way: " + "; ".join(star_ev) + "."))
    if agn_ev:
        cands.append(("agn_variability", min(0.95, 0.6 + 0.12 * len(agn_ev)),
                      "Known active galactic nucleus / quasar flickering: " + "; ".join(agn_ev) + "."))
    mag_now = t.get("mag_now")
    if host:
        nuclear = host["sep"] < max(0.6, 3 * (t.get("err") or 0.3))
        z = host.get("z")
        M = _abs_mag(mag_now, z)
        mtxt = f" At z = {z:.3f} the peak brightness {mag_now:.1f} mag means absolute magnitude about {M:.1f}" if M is not None else ""
        if M is not None:
            mtxt += (" (typical of a supernova)." if -21 < M < -15.5 else " (brighter than most supernovae: superluminous SN, TDE or AGN)."
                     if M <= -21 else " (fainter than most supernovae: nova, gap transient or a wrong host).")
        if nuclear:
            if not agn_ev:
                cands.append(("nuclear_transient", 0.55 + 0.1 * (host["d_dlr"] < 0.3),
                              f"Transient at the centre of its host galaxy ({host['sep']}\" from the nucleus): tidal disruption event, "
                              f"AGN flare or nuclear supernova.{mtxt}"))
        elif not star_ev:
            cands.append(("supernova_candidate", min(0.85, 0.5 + 0.4 * (1 - host["d_dlr"] / 4)),
                          f"Transient offset {host['sep']}\" ({host['d_dlr']} half-light radii) from a galaxy: the classic supernova "
                          f"configuration.{mtxt}"))
    arch = _archival_r(ident, rep)
    if arch and mag_now is not None:
        # alert magnitudes are difference-image fluxes (new light only), archival ones are total light
        f = photometry.flux_ratio(mag_now, arch["mag"])
        est = " (converted from Gaia G, estimated)" if arch["estimated"] else ""
        rep.claim("measurement", f"The changing light ({mag_now:.2f} mag in the difference image) is {f:.0%} of the archival source's total "
                                 f"brightness (r = {arch['mag']:.2f}{est}, {catalogs.LABELS[arch['survey']]}, epoch {arch['epoch'] or 0:.1f})"
                                 + (": a small flicker on top of a steady source." if f < 0.3 else
                                    ": a major brightening, the source at least doubled." if f >= 1 else "."),
                  arch["survey"], None, flux_ratio=round(f, 3), systematic_mag=arch["sys"])
    deep_enough = [n for n in POS_CATALOGS
                   if (rep.matches.get(n) or {}).get("status") == "covered" and coverage.should_have_seen(n, "r", mag_now)]
    nothing = not any((rep.matches.get(n) or {}).get("best") for n in POS_CATALOGS)
    if nothing and not host:
        if deep_enough:
            names = ", ".join(coverage.SURVEYS[n].label for n in deep_enough)
            cands.append(("new_hostless_transient", min(0.9, 0.5 + 0.15 * len(deep_enough)),
                          f"Nothing was ever seen here although {names} covered this position deeply enough to detect a "
                          f"{mag_now:.1f} mag source: this is evidence of absence. A genuinely new, hostless transient "
                          f"(or a host fainter than the catalogue limits)."))
        elif any((rep.matches.get(n) or {}).get("status") == "covered" for n in POS_CATALOGS):
            cands.append(("new_transient_unconstrained", 0.4,
                          "Nothing in the archives here, but the surveys that covered it are not deep enough to rule out a "
                          "faint quiescent source: absence of evidence, not evidence of absence."))
    known = [n for n in POS_CATALOGS if (rep.matches.get(n) or {}).get("best")]
    if known and not star_ev and not agn_ev and not host:
        cands.append(("known_source_variable", 0.45, "A catalogued source sits at this position ("
                      + ", ".join(catalogs.LABELS[n] for n in known) + ") but the archives do not say what it is: "
                      "variability of a known object of undetermined type."))
    if not cands:
        cov = [n for n in POS_CATALOGS if (rep.matches.get(n) or {}).get("status") == "covered"]
        cands.append(("insufficient_data", 0.3, "The archives do not settle what this is" +
                      (" (no survey we can query covered it)." if not cov else ".")))
    return cands


def _verdict_detection(rep, ident):
    """JWST / Hubble detections: do the archives agree with what our CNN says it is?"""
    t, cands = rep.target, []
    g, ls, sd = ident.get("gaia") or {}, ident.get("ls") or {}, ident.get("sdss") or {}
    star = ((g.get("plx_snr") or 0) > 5) or ((g.get("pm_snr") or 0) > 5 and (g.get("pm") or 0) > 3) or \
           ("*" in ((ident.get("simbad") or {}).get("otype") or "") and (ident.get("simbad") or {}).get("sep", 9) < 1)
    gal = ls.get("morph") == "extended" or (sd.get("morph") == "extended" and (sd.get("r") or 99) < 21.5) or (g.get("p_galaxy") or 0) > 0.5 or \
          ((ident.get("ned") or {}).get("otype") in ("G", "GPair", "GGroup"))
    qso = any(a in ((ident.get(k) or {}).get("otype") or "") for k in ("simbad", "ned") for a in AGN_TYPES) or (g.get("p_quasar") or 0) > 0.5
    if star and not gal:
        cands.append(("catalog_star", 0.9, "The archives say this is a Milky Way star (parallax / proper motion / SIMBAD)."))
    elif qso:
        cands.append(("catalog_qso", 0.85, "The archives list a quasar / AGN here."))
    elif gal:
        z = next((v for v in (sd.get("z"), (ident.get("ned") or {}).get("z"), (ident.get("simbad") or {}).get("z"), ls.get("z"))
                  if v is not None), None)
        cands.append(("catalog_galaxy", 0.85 if not star else 0.5, "The archives resolve this as a galaxy" +
                      (f" at redshift {z:.3f}" if z is not None else "") + "."))
        if star:
            rep.flags.append("inconsistent:star_vs_galaxy")
            rep.claim("inference", "Catalogues disagree: one sees a resolved galaxy, another measures stellar parallax or motion "
                                   "(probably two overlapping sources).", None, 0.5)
    elif ls.get("morph") == "point":
        cands.append(("catalog_point_source", 0.6, "Point-like in the Legacy Surveys but without parallax or motion: "
                                                   "a distant star, compact galaxy or quasar."))
    else:
        cands.append(("archives_too_shallow", 0.5, "No catalogue counterpart: Hubble / JWST see far fainter than the archive surveys, "
                                                   "so silence here is expected, not suspicious."))
    ours = t.get("cnn_label") or t.get("label")
    if ours:
        src = "Our CNN" if t.get("cnn_label") else "Our detector"
        ours_gal = ours in GALAXY_LABELS or "galaxy" in ours.lower() and ours != "Not a galaxy"
        v = cands[0][0]
        if v in ("catalog_star", "catalog_galaxy", "catalog_qso"):
            agree = (v == "catalog_galaxy") == ours_gal
            conf_txt = f" ({t['cnn_conf']:.0%})" if t.get("cnn_conf") else ""
            rep.claim("model_prediction", f"{src} called it '{ours}'{conf_txt}; the archives {'agree' if agree else 'DISAGREE'}.",
                      "galaxy_cnn" if t.get("cnn_label") else "detector", t.get("cnn_conf"), agree=agree)
            rep.target["cnn_agrees"] = agree
    return cands


def _moving(rep, host):
    """Ask JPL which asteroids were at this spot at the alert time, when the alert could be a moving object.
    (It is slow, ~1 minute, so only for single detections, broker 'asteroid' labels, or hostless positions.)"""
    t = rep.target
    if t["kind"] != "alert" or not t.get("mjd"):
        return
    nothing = not any((rep.matches.get(n) or {}).get("best") for n in POS_CATALOGS)
    if not ("asteroid" in (t.get("classifier_cls") or "").lower() or (t.get("n_det") or 9) <= 1 or (nothing and not host)):
        return
    res = catalogs.jpl_sbident(t["ra"], t["dec"], t["mjd"], t.get("survey", "lsst"))
    if not res["ok"]:
        rep.claim("missing", f"JPL asteroid check failed ({res.get('error')}); a moving object is not ruled out.", "jpl_sbident")
        return
    lim = max(10.0, 5 * (t.get("err") or 0.3))
    rep.asteroids = [b for b in res["sources"] if b["sep"] <= lim]
    if rep.asteroids:
        for b in rep.asteroids:
            rep.claim("catalog_fact", f"JPL ephemeris: {b['name']} predicted {b['sep']:.1f}\" away at the alert time (V ≈ {b['V']}).",
                      "jpl_sbident", None, name=b["name"])
    else:
        rep.claim("catalog_fact", f"No known asteroid or comet within {lim:.0f}\" at the alert time (JPL ephemerides).", "jpl_sbident")


def analyze(target, con=None, persist=True, cats=None):
    """target: {kind, ref, ra, dec, err (arcsec), epoch (yr), ...}. Returns the evidence report (dict).
    cats: catalogue results already fetched for a whole field (analyze_image), else queried here."""
    t0 = time.time()
    rep = Report(target)
    coarse = (target.get("err") or 0) > 30
    cats = cats or catalogs.query_all(target["ra"], target["dec"])
    if coarse:
        rep.claim("inference", f"Localisation is only ±{target['err']:.0f}\": too coarse to name a single counterpart; "
                               "catalogue sources at the centre are context, not identifications.", None, None)
    _counterparts(rep, cats)
    ident = _identity(rep)
    host = _host(rep, cats) if target["kind"] != "detection" else None
    _history(rep, cats, con)
    _moving(rep, host)
    if target.get("classifier_cls"):
        rep.claim("model_prediction", f"{target.get('classifier') or 'Broker'} classifier: {target['classifier_cls']}"
                                      f" (p = {target.get('classifier_p') or 0:.2f}).", target.get("classifier"), target.get("classifier_p"))
    cands = _verdict_detection(rep, ident) if target["kind"] == "detection" else _verdict_transient(rep, ident, host)
    if coarse:
        cands = [("coarse_localisation", 0.2, "Position too uncertain for a cross-survey identification.")]
    cands.sort(key=lambda c: -c[1])
    verdict, conf, why = cands[0]
    if "bogus" in (target.get("classifier_cls") or "").lower() and verdict != "known_asteroid":
        conf *= 0.7
        rep.flags.append("broker_bogus")
        rep.claim("inference", "The broker's stamp classifier thinks this alert is an artefact (bogus); every verdict here is "
                               "conditional on it being real.", target.get("classifier"), target.get("classifier_p"))
    failed = [n for n in POS_CATALOGS if (rep.matches.get(n) or {}).get("status") == "failed"]
    conf *= 0.85 ** len(failed)
    if any(f.startswith(("blend", "inconsistent")) for f in rep.flags):
        conf *= 0.85
    rep.claim("inference", why, "xsurvey", conf, verdict=verdict, alternatives=[{"verdict": c[0], "confidence": round(c[1], 3)} for c in cands[1:]])
    out = {"ref": target["ref"], "kind": target["kind"], "ra": target["ra"], "dec": target["dec"], "err_arcsec": target.get("err"),
           "verdict": verdict, "confidence": round(conf, 3), "summary": why, "flags": rep.flags,
           "alternatives": [{"verdict": c[0], "confidence": round(c[1], 3), "why": c[2]} for c in cands[1:]],
           "identity": ident, "host": host, "evidence": rep.evidence, "history": rep.history,
           "coverage": {n: (rep.matches.get(n) or {}).get("status") for n in POS_CATALOGS},
           "matches": {n: {k: v for k, v in (i or {}).items() if k != "best"} for n, i in rep.matches.items()},
           "cnn_agrees": target.get("cnn_agrees"),
           "services": {n: {"ok": r["ok"], "latency_s": r.get("latency_s"), "error": r.get("error")} for n, r in cats.items()},
           "engine": ENGINE_VERSION, "elapsed_s": round(time.time() - t0, 2), "created_at": _now()}
    if persist and con is not None:
        save(con, out)
    return out


# ---------- persistence ----------

def save(con, rep):
    now = _now()
    con.execute("""INSERT INTO astronomical_objects (kind, ref, ra, dec, pos_err_arcsec, epoch, verdict, verdict_conf, report, created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(ref) DO UPDATE SET ra=excluded.ra, dec=excluded.dec, pos_err_arcsec=excluded.pos_err_arcsec,
                     verdict=excluded.verdict, verdict_conf=excluded.verdict_conf, report=excluded.report, updated_at=excluded.updated_at""",
                (rep["kind"], rep["ref"], rep["ra"], rep["dec"], rep["err_arcsec"], None, rep["verdict"], rep["confidence"],
                 json.dumps(rep, default=str), now, now))
    oid = con.execute("SELECT id FROM astronomical_objects WHERE ref=?", (rep["ref"],)).fetchone()[0]
    for tbl in ("cross_survey_matches", "historical_observations", "scientific_evidence"):
        con.execute(f"DELETE FROM {tbl} WHERE object_id=?", (oid,))
    con.executemany("""INSERT INTO cross_survey_matches (object_id, catalog, cat_id, ra, dec, sep_arcsec, p_match, rank, ambiguous, pm_corrected, data)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                    [(oid, n, str(m.get("id")), m["ra"], m["dec"], m["sep_arcsec"], m["p_match"], m["rank"], int(m["ambiguous"]),
                      int(m["pm_corrected"]), json.dumps({k: m.get(k) for k in ("mags", "morph", "redshift", "otype", "name", "parallax")}))
                     for n, i in rep["matches"].items() for m in (i.get("matches") or [])[:3]])
    con.executemany("""INSERT INTO historical_observations (object_id, survey, epoch, mjd, band, mag, mag_err, upper_limit, estimated, source)
                       VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    [(oid, h["survey"], h.get("epoch"), h.get("mjd"), h.get("band"), h.get("mag"), h.get("mag_err"), h["upper_limit"],
                      h["estimated"], h["source"]) for h in rep["history"]])
    con.executemany("INSERT INTO scientific_evidence (object_id, claim_type, statement, source, confidence, data, created_at) VALUES (?,?,?,?,?,?,?)",
                    [(oid, e["claim_type"], e["statement"], e["source"], e["confidence"], json.dumps(e["data"], default=str), now)
                     for e in rep["evidence"]])
    con.execute("INSERT OR IGNORE INTO model_versions (name, version, created_at, notes) VALUES (?,?,?,?)",
                ("xsurvey", ENGINE_VERSION, now, "Bayesian positional match + coverage-aware rule verdicts"))
    con.commit()
    return oid


def load(con, ref):
    r = con.execute("SELECT report FROM astronomical_objects WHERE ref=?", (ref,)).fetchone()
    return json.loads(r[0]) if r and r[0] else None


# ---------- targets from what the project already stores ----------

def alert_target(con, survey, oid, with_lc=True):
    a = con.execute("SELECT * FROM alerts WHERE survey=? AND oid=?", (survey, str(oid))).fetchone()
    if not a or a["ra"] is None:
        return None
    t = {"kind": "alert", "ref": f"alert:{survey}:{oid}", "survey": survey, "oid": str(oid), "ra": a["ra"], "dec": a["dec"],
         "err": 0.1 if survey == "lsst" else 0.3, "epoch": mjd_to_year(a["lastmjd"]) if a["lastmjd"] else None,
         "mjd": a["lastmjd"], "n_det": a["n_det"],
         "classifier": a["classifier"], "classifier_cls": a["cls"], "classifier_p": a["probability"]}
    if with_lc:
        try:
            from ..context import light_curve
            pts = light_curve(survey, oid).get("points") or []
        except Exception as e:  # noqa: BLE001
            log.info("light curve for %s unavailable: %s", oid, e)
            pts = []
        rl = [p for p in pts if p["band"] in ("r", "R", "ztf_r", "g", "i")] or pts
        if rl:
            pk = min(rl, key=lambda p: p["mag"])
            t.update(mag_now=pk["mag"], mag_err=pk.get("err"), band=pk["band"], light_curve=pts)
    return t


def detection_target(con, det_id):
    d = con.execute("""SELECT d.*, i.source, i.obs_time, i.target AS field FROM detections d JOIN images i ON i.id = d.image_id
                       WHERE d.id=?""", (det_id,)).fetchone()
    if not d or d["ra"] is None:
        return None
    yr = int(d["obs_time"][:4]) + (int(d["obs_time"][5:7]) - 0.5) / 12 if d["obs_time"] else None
    return {"kind": "detection", "ref": f"detection:{d['image_id']}:{det_id}", "ra": d["ra"], "dec": d["dec"], "err": 0.3, "epoch": yr,
            "label": d["label"], "cnn_label": d["cnn_label"], "cnn_conf": d["cnn_conf"], "field": d["field"]}


def gcn_target(ev):
    if ev.get("ra") is None:
        return None
    subj = " ".join(c["subject"] for c in ev.get("circulars", [])).lower()
    err = 1.0 if any(k in subj for k in ("optical", "uvot", "afterglow", "nir", "spectroscop")) else 3.0 if "xrt" in subj or "x-ray" in subj else 60.0
    return {"kind": "gcn", "ref": f"gcn:{ev['event']}", "ra": ev["ra"], "dec": ev["dec"], "err": err, "epoch": None}


def analyze_image(con, image_id, limit=None, min_score=0.0):
    """Every located detection of one JWST/HST image against ONE catalogue query covering the field.
    Returns {n, verdicts, cnn_agreement, ...}."""
    from collections import Counter
    rows = con.execute("SELECT id, ra, dec FROM detections WHERE image_id=? AND ra IS NOT NULL AND score >= ? ORDER BY score DESC"
                       + (f" LIMIT {int(limit)}" if limit else ""), (image_id, min_score)).fetchall()
    if not rows:
        return {"image_id": image_id, "n": 0}
    ras = [r["ra"] for r in rows]
    decs = [r["dec"] for r in rows]
    ra0 = (sum(((a - ras[0] + 180) % 360 - 180) for a in ras) / len(ras) + ras[0]) % 360
    dec0 = sum(decs) / len(decs)
    radius = min(max(sep_arcsec(ra0, dec0, a, d) for a, d in zip(ras, decs)) + CAND_RADIUS + 5, 900)
    cats = catalogs.query_all(ra0, dec0, only=POS_CATALOGS + NAME_CATALOGS, radius=max(radius, catalogs.FIELD_RADIUS + 1))
    verdicts, agree = Counter(), Counter()
    for r in rows:
        t = detection_target(con, r["id"])
        rep = analyze(t, con, cats=cats)
        verdicts[rep["verdict"]] += 1
        if rep.get("cnn_agrees") is not None:
            agree["agree" if rep["cnn_agrees"] else "disagree"] += 1
    n = agree["agree"] + agree["disagree"]
    return {"image_id": image_id, "n": len(rows), "field_radius_arcsec": round(radius, 1), "verdicts": dict(verdicts),
            "cnn_checked": n, "cnn_agreement": round(agree["agree"] / n, 3) if n else None,
            "services": {k: v["ok"] for k, v in cats.items()}}
