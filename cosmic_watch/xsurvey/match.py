"""Probabilistic positional cross-matching (pure functions, no network).

For a target position and a list of catalogue candidates, each candidate gets a posterior probability
of being the same source:

    L_same   = exp(-s^2 / 2 sigma^2) / (2 pi sigma^2)      Gaussian on the separation s, sigma^2 = sigma_t^2 + sigma_c^2
    L_chance = rho                                          density of unrelated sources around the target (per arcsec^2)
    P        = prior * L_same / (prior * L_same + (1 - prior) * L_chance)

The density comes from the catalogue's own source count in the query cone, so a 1" offset is convincing in an
empty high-latitude field and much less so in a crowded field near the Galactic plane. This is the
single-catalogue form of the Bayesian cross-match of Budavari & Szalay (2008).

Proper motion: catalogue positions with a proper motion (Gaia) are moved to the target's epoch before matching,
and their positional uncertainty grows with the proper-motion error times the epoch difference.
"""
import math

MAS = 1 / 3_600_000                     # degrees per milliarcsecond
DEFAULT_PRIOR = 0.5                     # P(a counterpart exists at all) before looking
AMBIGUOUS_P = 0.2                       # two candidates above this = blended / ambiguous


def sep_arcsec(ra1, dec1, ra2, dec2):
    """Great-circle separation (haversine, accurate at small angles)."""
    r1, d1, r2, d2 = map(math.radians, (ra1, dec1, ra2, dec2))
    h = math.sin((d2 - d1) / 2) ** 2 + math.cos(d1) * math.cos(d2) * math.sin((r2 - r1) / 2) ** 2
    return math.degrees(2 * math.asin(min(1.0, math.sqrt(h)))) * 3600


def propagate(ra, dec, pmra, pmdec, epoch_from, epoch_to):
    """Linear proper-motion propagation. pmra is mu_alpha* (includes cos dec), mas/yr. Returns (ra, dec)."""
    if pmra is None or pmdec is None or epoch_from is None or epoch_to is None:
        return ra, dec
    dt = epoch_to - epoch_from
    ddec = pmdec * dt * MAS
    dra = pmra * dt * MAS / max(math.cos(math.radians(dec)), 1e-6)
    return (ra + dra) % 360, dec + ddec


def match(target, candidates, field_count=None, field_radius_arcsec=None, prior=DEFAULT_PRIOR):
    """target: {ra, dec, err (arcsec, 1-sigma), epoch}
    candidates: [{ra, dec, err, epoch?, pmra?, pmdec?, pmra_err?, pmdec_err?, ...}]
    field_count / field_radius: how many sources the catalogue has in a cone (for the chance density).
    Returns candidates sorted by p_match, each with sep_arcsec, sigma_arcsec, p_match, rank, pm_corrected."""
    out = []
    n = field_count if field_count is not None else len(candidates)
    rad = field_radius_arcsec or max([c.get("sep_hint", 0) for c in candidates] + [10.0])
    rho = max(n, 1) / (math.pi * rad ** 2)                          # sources per arcsec^2
    for c in candidates:
        c = dict(c)
        ra, dec, err = c["ra"], c["dec"], c.get("err") or 0.1
        corrected = False
        if c.get("pmra") is not None and c.get("epoch") is not None and target.get("epoch") is not None:
            dt = abs(target["epoch"] - c["epoch"])
            ra, dec = propagate(ra, dec, c["pmra"], c["pmdec"], c["epoch"], target["epoch"])
            pm_err = math.hypot(c.get("pmra_err") or 0, c.get("pmdec_err") or 0) * dt / 1000
            err = math.hypot(err, pm_err)
            corrected = True
        s = sep_arcsec(target["ra"], target["dec"], ra, dec)
        sig2 = (target.get("err") or 0.1) ** 2 + err ** 2
        l_same = math.exp(-s * s / (2 * sig2)) / (2 * math.pi * sig2)
        p = prior * l_same / (prior * l_same + (1 - prior) * rho)
        c.update(sep_arcsec=round(s, 3), sigma_arcsec=round(math.sqrt(sig2), 3), p_match=round(p, 4),
                 pm_corrected=corrected, ra_at_epoch=ra, dec_at_epoch=dec)
        out.append(c)
    out.sort(key=lambda c: -c["p_match"])
    for i, c in enumerate(out):
        c["rank"] = i + 1
    ambiguous = sum(1 for c in out if c["p_match"] >= AMBIGUOUS_P) > 1
    for c in out:
        c["ambiguous"] = ambiguous
    return out


def best(matches, min_p=0.5):
    return matches[0] if matches and matches[0]["p_match"] >= min_p else None
