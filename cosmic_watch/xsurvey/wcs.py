"""Sky positions for JWST / Hubble detections, from the science mosaic's WCS.

Detections are found on MAST's JPEG preview, which has no coordinates. The science mosaic (*_i2d / *_drc) has the
same footprint and carries a WCS in its SCI header, so we fetch only the first ~300 kB of the FITS file with an HTTP
range request (headers only, never the gigabytes of pixels), build an astropy WCS and map each detection:

    preview (x, y), y down  ->  science pixel (x * NAXIS1 / W, (H - y) * NAXIS2 / H), FITS rows bottom-up
    science pixel           ->  (RA, Dec) via the WCS (includes SIP distortion when the header has it)

The preview and mosaic are the same grid up to scale (the deep-zoom science upgrade relies on the same fact).
Headers are cached as text so a re-run never touches the network.
"""
import logging
import math

import numpy as np
import requests

from ..config import DATA

log = logging.getLogger("cosmic_watch")
HDR_DIR = DATA / "xsurvey" / "wcs"
MAST_FILE = "https://mast.stsci.edu/api/v0.1/Download/file"
HEAD_BYTES = 400_000
BLOCK = 2880


def _uris(source, external_id):
    if source == "jwst":
        return [f"mast:JWST/product/{external_id}_i2d.fits"]
    return [f"mast:HST/product/{external_id}_drc.fits", f"mast:HST/product/{external_id}_drz.fits"]


def parse_headers(buf):
    """Walk FITS HDUs in a byte prefix; return [(Header)] for every header fully contained in it."""
    from astropy.io import fits
    out, pos = [], 0
    while pos + BLOCK <= len(buf):
        end = None
        for b in range(pos, len(buf) - BLOCK + 1, BLOCK):
            block = buf[b:b + BLOCK]
            for i in range(0, BLOCK, 80):
                if block[i:i + 8] == b"END     ":
                    end = b + BLOCK
                    break
            if end:
                break
        if not end:
            break
        h = fits.Header.fromstring(buf[pos:end].decode("ascii", "replace"))
        out.append(h)
        n = 1
        for k in range(1, h.get("NAXIS", 0) + 1):
            n *= h.get(f"NAXIS{k}", 0)
        nbytes = abs(h.get("BITPIX", 8)) // 8 * (h.get("GCOUNT", 1) * (h.get("PCOUNT", 0) + n) if h.get("NAXIS", 0) else 0)
        pos = end + math.ceil(nbytes / BLOCK) * BLOCK
    return out


def sci_header(source, external_id):
    """The SCI extension header (cached), or None if MAST has no science mosaic for this preview."""
    from astropy.io import fits
    p = HDR_DIR / f"{external_id}.hdr"
    if p.exists():
        txt = p.read_text(encoding="ascii")
        return fits.Header.fromstring(txt) if txt else None
    for uri in _uris(source, external_id):
        try:
            r = requests.get(MAST_FILE, params={"uri": uri}, headers={"Range": f"bytes=0-{HEAD_BYTES - 1}"}, timeout=60, stream=True)
            if r.status_code not in (200, 206):
                r.close()
                continue
            buf = r.raw.read(HEAD_BYTES)
            r.close()
        except requests.RequestException as e:
            log.warning("WCS header fetch failed for %s: %s", external_id, e)
            return None                                         # network: retry another time, cache nothing
        for h in parse_headers(buf):
            if h.get("EXTNAME") == "SCI" or ("CTYPE1" in h and h.get("NAXIS", 0) == 2):
                HDR_DIR.mkdir(parents=True, exist_ok=True)
                p.write_text(h.tostring(), encoding="ascii")
                return h
    HDR_DIR.mkdir(parents=True, exist_ok=True)
    p.write_text("", encoding="ascii")                         # no mosaic: remember, don't retry
    return None


def preview_to_sky(header, preview_size, xy):
    """xy: array (N, 2) of preview pixels (x right, y down). Returns (ra, dec) arrays in degrees."""
    import warnings

    from astropy.wcs import WCS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w = WCS(header)
    W, H = preview_size
    nx, ny = header["NAXIS1"], header["NAXIS2"]
    xy = np.asarray(xy, dtype=float)
    px = xy[:, 0] * nx / W - 0.5 * (nx / W - 1)
    py = (H - xy[:, 1]) * ny / H - 0.5 * (ny / H - 1)
    ra, dec = w.all_pix2world(px, py, 0)
    return ra, dec


def pixel_scale_arcsec(header, preview_size):
    """Arcsec per *preview* pixel (for turning detection sizes and positional errors into angles)."""
    import warnings

    from astropy.wcs import WCS
    from astropy.wcs.utils import proj_plane_pixel_scales
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = float(np.mean(proj_plane_pixel_scales(WCS(header).celestial))) * 3600
    return s * header["NAXIS1"] / preview_size[0]


def locate_image(con, image_id):
    """Fill detections.ra/dec for one JWST/HST image. Returns number located (0 if no WCS)."""
    from PIL import Image

    from ..config import ROOT
    r = con.execute("SELECT source, external_id, path FROM images WHERE id=?", (image_id,)).fetchone()
    if not r or r["source"] not in ("jwst", "hst"):
        return 0
    h = sci_header(r["source"], r["external_id"])
    if h is None:
        return 0
    path = ROOT / r["path"] if not str(r["path"]).startswith(("/", "\\")) and ":" not in str(r["path"]) else r["path"]
    with Image.open(path) as im:
        size = im.size
    rows = con.execute("SELECT id, x, y FROM detections WHERE image_id=? AND ra IS NULL", (image_id,)).fetchall()
    if not rows:
        return 0
    ra, dec = preview_to_sky(h, size, [(d["x"], d["y"]) for d in rows])
    con.executemany("UPDATE detections SET ra=?, dec=? WHERE id=?",
                    [(float(a), float(b), d["id"]) for a, b, d in zip(ra, dec, rows) if np.isfinite(a) and np.isfinite(b)])
    con.commit()
    return len(rows)
