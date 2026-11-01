"""Comparing brightness across surveys (pure functions).

Different telescopes use different filters and magnitude systems. To compare them we bring everything to an
approximate "r-band AB" scale and carry a systematic uncertainty for the conversion, and every converted value
is flagged as estimated:

* PS1 r, SDSS r, Legacy Survey r, ZTF r, LSST r: all close to AB r; small offsets (< 0.05 mag) are folded
  into a 0.05 mag systematic.
* ZTF g, PS1 g...: kept in their own band; compared only like with like.
* Gaia G: converted to SDSS r with the Gaia DR3 colour relation (Riello et al. 2021) when BP-RP is known,
  otherwise G - r ~ -0.1 with a 0.3 mag systematic.
* Rubin alert fluxes (nJy): m_AB = 31.4 - 2.5 log10(f).
"""
import math

R_LIKE = {"r", "R", "rp1", "r_sdss", "r_ls", "ztf_r"}
SYS = {"same": 0.05, "gaia_color": 0.12, "gaia_nocolor": 0.3}


def njy_to_ab(flux, err=None):
    if flux is None or flux <= 0:
        return None, None
    m = 31.4 - 2.5 * math.log10(flux)
    return m, (1.0857 * err / flux if err else None)


def gaia_g_to_r(g, bp_rp=None):
    """Gaia G -> SDSS r (Gaia DR3 documentation, photometric relations; valid ~ -0.5 < BP-RP < 4)."""
    if g is None:
        return None, None
    if bp_rp is None or not (-0.5 < bp_rp < 4):
        return g + 0.1, SYS["gaia_nocolor"]
    x = bp_rp
    g_minus_r = -0.09837 + 0.08592 * x + 0.1907 * x ** 2 - 0.1701 * x ** 3 + 0.02263 * x ** 4
    return g - g_minus_r, SYS["gaia_color"]


def compare(m1, e1, s1, m2, e2, s2, n_sigma=3.0, floor=0.2):
    """Is the brightness change between two measurements significant?
    s1/s2 = systematic errors of each value's conversion. Returns (delta, significant, sigma)."""
    if m1 is None or m2 is None:
        return None, False, None
    sig = math.sqrt((e1 or 0.05) ** 2 + (e2 or 0.05) ** 2 + (s1 or 0) ** 2 + (s2 or 0) ** 2)
    d = m2 - m1
    return round(d, 3), abs(d) > max(n_sigma * sig, floor), round(sig, 3)


def flux_ratio(m_new, m_ref):
    """Flux of m_new as a fraction of m_ref (e.g. difference-image flux vs a catalogue's total flux)."""
    return 10 ** (-0.4 * (m_new - m_ref))
