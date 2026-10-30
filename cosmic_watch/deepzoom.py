"""Deep zoom: tiled image pyramids, so zooming in never stretches pixels.

Two kinds of pyramid, both served as 512 px tiles (OpenSeadragon level numbering:
level L shows the image at scale 2^(L - max_level)):

* JWST / Hubble images we stored at full resolution: tiles are cut from the full-res file.
* The sky around an alert: tiles are requested from the DESI Legacy Surveys cut-out service
  (DECaLS / BASS / MzLS imaging, 0.262"/px native) at the right position and pixel scale
  for each tile, and cached on disk. Rubin's alert stamps are only tens of pixels across;
  this gives a deep, sharp map of the neighbourhood to place them on.

The host galaxy of an alert is found with SEP on the native-resolution survey cut-out and
classified by our CNN; Galaxy10 itself was made from this same survey, so there is no
domain gap.
"""
import json
import logging
import math
import threading
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import requests
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from PIL import Image

from . import db
from .config import DATA, ROOT
from .ml.classes import CLASSES, COARSE

log = logging.getLogger("cosmic_watch")
router = APIRouter()

TILE = 512
TILE_DIR = DATA / "tiles"
SCIENCE_DIR = DATA / "science"
JPEG_Q = 90
SCIENCE_Q = 95            # compressed only after stretching, so faint sky keeps its real grain
IMG_TILE_VERSION = 3      # bump when tile processing changes (old tiles are then ignored)
MAST_FILE = "https://mast.stsci.edu/api/v0.1/Download/file"

# ---------- JWST / Hubble pyramids ----------
#
# Tiles start from MAST's JPEG preview (instant), and upgrade to the science FITS mosaic
# (JWST *_i2d.fits / Hubble *_drc.fits, float32) in the background the first time someone
# deep-zooms an image. Same pixel grid, but no JPEG blocks: at high zoom the faint sky shows
# its true grain instead of 8x8 compression artifacts amplified by the stretch.

_pyr: "OrderedDict[int, dict]" = OrderedDict()   # image id -> {"full": array, levels...}
_pyr_lock = threading.Lock()
PYR_KEEP = 3                                      # full-res images held in memory
_science: dict[int, dict] = {}                    # image id -> {"state", "progress", "error"}
_science_lock = threading.Semaphore(1)            # one big download at a time


def _stretch_lut(sky_pixels, soften=12.0):
    """Display stretch (asinh) from the image's own sky level to its bright end.

    MAST previews are close to linear, so faint galaxies sit a few counts above black.
    This only remaps brightness (monotonic, per pixel); it adds no detail.
    """
    lo, hi = np.percentile(sky_pixels, [35, 99.7]) if sky_pixels.size else (0, 255)
    hi = max(hi, lo + 8)
    x = np.clip((np.arange(256, dtype=np.float32) - lo) / (hi - lo), 0, None)
    y = np.arcsinh(x * soften) / np.arcsinh(soften)
    return np.clip(y * 255, 0, 255).astype(np.uint8)


def _stretch_science(sci, soften=12.0):
    """float32 science array -> uint8 asinh stretch from (sky - 1 sigma) to the 99.8th percentile."""
    ok = np.isfinite(sci) & (sci != 0)
    vals = sci[ok]
    if vals.size > 4_000_000:
        vals = vals[:: vals.size // 4_000_000]
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med))) * 1.4826
    lo, hi = med - mad, float(np.percentile(vals, 99.8))
    hi = max(hi, lo + 1e-6)
    x = np.clip((np.where(ok, sci, lo) - lo) / (hi - lo), 0, None)
    img = (np.clip(np.arcsinh(x * soften) / np.arcsinh(soften), 0, 1) * 255).astype(np.uint8)
    img[~ok] = 0
    return img


def _levels(w, h):
    return math.ceil(math.log2(max(w, h)))


def _image_row(image_id):
    con = db.connect()
    try:
        r = con.execute("SELECT id, source, external_id, path FROM images WHERE id=?", (image_id,)).fetchone()
    finally:
        con.close()
    if not r:
        raise HTTPException(404)
    p = Path(r["path"])
    p = p if p.is_absolute() else ROOT / p
    if not p.is_file():
        raise HTTPException(404)
    return r, p


def _science_png(image_id) -> Path:
    return SCIENCE_DIR / f"{image_id}.png"


def _science_status(image_id, source):
    if source not in ("jwst", "hst"):
        return {"state": "unavailable"}
    if _science_png(image_id).exists():
        return {"state": "ready"}
    return _science.get(image_id, {"state": "available"})


def _science_uris(r):
    ext = r["external_id"]
    if r["source"] == "jwst":
        return [f"mast:JWST/product/{ext}_i2d.fits"]
    return [f"mast:HST/product/{ext}_drc.fits", f"mast:HST/product/{ext}_drz.fits"]


def _fetch_science(image_id):
    """Background: download the science mosaic, stretch it, keep a lossless PNG, drop the FITS."""
    from astropy.io import fits
    st = _science[image_id]
    with _science_lock:
        r, p = _image_row(image_id)
        SCIENCE_DIR.mkdir(parents=True, exist_ok=True)
        part = SCIENCE_DIR / f"{image_id}.fits.part"
        try:
            for uri in _science_uris(r):
                with _http.get(MAST_FILE, params={"uri": uri}, stream=True, timeout=60) as resp:
                    if resp.status_code != 200:
                        continue
                    total = int(resp.headers.get("Content-Length") or 0)
                    got = 0
                    with open(part, "wb") as f:
                        for chunk in resp.iter_content(1 << 20):
                            f.write(chunk)
                            got += len(chunk)
                            if total:
                                st["progress"] = round(0.95 * got / total, 3)
                    break
            else:
                raise RuntimeError("no science mosaic on MAST for this preview")
            st["state"], st["progress"] = "processing", 0.96
            with fits.open(part, memmap=True) as hdul:
                sci = np.asarray(hdul["SCI"].data, dtype=np.float32)
            img = _stretch_science(sci)[::-1]          # FITS rows run bottom-up
            with Image.open(p) as prev:
                if prev.size != (img.shape[1], img.shape[0]):   # keep the detections' pixel grid
                    img = cv2.resize(img, prev.size, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(_science_png(image_id)), np.ascontiguousarray(img))
            with _pyr_lock:
                _pyr.pop(image_id, None)
            st.update(state="ready", progress=1.0)
            log.info("science data ready for image %s (%s)", image_id, r["external_id"])
        except Exception as e:  # noqa: BLE001
            log.warning("science fetch for image %s failed: %s", image_id, e)
            st.update(state="failed", error=str(e)[:200])
        finally:
            part.unlink(missing_ok=True)


def _pyramid(image_id):
    with _pyr_lock:
        if image_id in _pyr:
            _pyr.move_to_end(image_id)
            return _pyr[image_id]
    r, p = _image_row(image_id)
    sci = _science_png(image_id)
    if sci.exists():
        bgr = cv2.cvtColor(cv2.imread(str(sci), cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2BGR)
        kind = "science"
    else:
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            raise HTTPException(404)
        kind = "preview"
        if r["source"] in ("jwst", "hst"):
            # MAST pads outside the footprint with flat white: make it black, like the thumbnails
            from .detectors.galaxies import _footprint_mask
            outside = _footprint_mask(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32))
            bgr = cv2.LUT(bgr, _stretch_lut(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)[~outside]))
            bgr[outside] = 0
    entry = {"full": bgr, "max": _levels(bgr.shape[1], bgr.shape[0]), "scaled": {}, "kind": kind}
    with _pyr_lock:
        _pyr[image_id] = entry
        while len(_pyr) > PYR_KEEP:
            _pyr.popitem(last=False)
    return entry


def _level_image(entry, level):
    if level >= entry["max"]:
        return entry["full"]
    if level not in entry["scaled"]:
        f = 2.0 ** (level - entry["max"])
        h, w = entry["full"].shape[:2]
        entry["scaled"][level] = cv2.resize(entry["full"], (max(1, round(w * f)), max(1, round(h * f))),
                                            interpolation=cv2.INTER_AREA)
    return entry["scaled"][level]


@router.get("/api/dz/image/{image_id}")
def image_info(image_id: int):
    """Tile-source description for one stored image (full-resolution pixel size)."""
    r, p = _image_row(image_id)
    with Image.open(p) as im:       # reads the header only
        w, h = im.size
    sci = _science_status(image_id, r["source"])
    kind = "science" if sci["state"] == "ready" else "preview"
    return {"width": w, "height": h, "tileSize": TILE, "tileOverlap": 1, "maxLevel": _levels(w, h),
            "source": r["source"], "data": kind, "science": sci,
            "ai": {"available": sr_available(), "models": {k: {"name": v["name"], "note": v["note"]} for k, v in SR_MODELS.items()},
                   "tiles": f"/api/dz/image/{image_id}/ai/{{model}}/{{x}}_{{y}}.jpg?v={IMG_TILE_VERSION}&d={kind}"},
            "tiles": f"/api/dz/image/{image_id}/{{level}}/{{x}}_{{y}}.jpg?v={IMG_TILE_VERSION}&d={kind}"}


@router.post("/api/dz/image/{image_id}/science")
def image_science(image_id: int):
    """Start (once) the upgrade from the JPEG preview to the science FITS; returns its status."""
    r, _ = _image_row(image_id)
    st = _science_status(image_id, r["source"])
    if st["state"] in ("available", "failed"):
        st = _science[image_id] = {"state": "fetching", "progress": 0.0}
        threading.Thread(target=_fetch_science, args=(image_id,), daemon=True).start()
    return st


@router.get("/api/dz/image/{image_id}/{level}/{xy}.jpg")
def image_tile(image_id: int, level: int, xy: str, d: str = "preview"):
    entry = _pyramid(image_id)
    out = TILE_DIR / f"img-v{IMG_TILE_VERSION}" / entry["kind"] / str(image_id) / str(level) / f"{xy}.jpg"
    if not out.exists():
        try:
            x, y = (int(v) for v in xy.split("_"))
        except ValueError:
            raise HTTPException(400)
        if level < 0 or level > entry["max"]:
            raise HTTPException(404)
        im = _level_image(entry, level)
        h, w = im.shape[:2]
        x0, y0 = x * TILE - (1 if x else 0), y * TILE - (1 if y else 0)     # 1 px overlap hides seams
        x1, y1 = min(w, (x + 1) * TILE + 1), min(h, (y + 1) * TILE + 1)
        if x0 >= w or y0 >= h:
            raise HTTPException(404)
        out.parent.mkdir(parents=True, exist_ok=True)
        q = SCIENCE_Q if entry["kind"] == "science" else JPEG_Q
        cv2.imwrite(str(out), im[y0:y1, x0:x1], [cv2.IMWRITE_JPEG_QUALITY, q])
    return FileResponse(out, headers={"Cache-Control": "public, max-age=604800"})


# ---------- AI enhance: one extra 2x level made by a super-resolution network ----------
#
# Past the real resolution the viewer can only interpolate. With AI enhance on, the pyramid
# gets one more level (2x the real pixel count) whose tiles come from a super-resolution
# model. That detail is *invented*: plausible, not measured. The viewer labels it, and it
# never feeds the CNN, the detectors or the anomaly scores.

SR_DIR = DATA / "models" / "sr"
SR_MODELS = {
    "swinir": {"file": "swinir_m_x2.pth", "name": "SwinIR ×2",
               "url": "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth",
               "note": "sharper; stays close to the data", "fp16": False},   # overflows in half precision
    "esrgan": {"file": "realesrgan_x2plus.pth", "name": "Real-ESRGAN ×2",
               "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
               "note": "cleaner, but smooths away faint sources", "fp16": True},
}
SR_PAD = 16              # input context around each tile, so tile edges match
_sr: dict[str, tuple] = {}
_sr_lock = threading.Lock()


def sr_available() -> bool:
    try:
        import spandrel  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def _sr_model(key):
    if key not in _sr:
        import torch
        from spandrel import ModelLoader
        cfg = SR_MODELS[key]
        path = SR_DIR / cfg["file"]
        if not path.exists():
            SR_DIR.mkdir(parents=True, exist_ok=True)
            part = path.with_suffix(".part")
            with _http.get(cfg["url"], stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(part, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            part.replace(path)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _sr[key] = (ModelLoader().load_from_file(str(path)).to(dev).eval(), dev)
    return _sr[key]


def _super_resolve(gray, key):
    """uint8 grey (H, W) -> uint8 grey (2H, 2W)."""
    import torch
    model, dev = _sr_model(key)
    h, w = gray.shape
    ph, pw = (-h) % 8, (-w) % 8                    # window / stride multiples
    g = cv2.copyMakeBorder(gray, 0, ph, 0, pw, cv2.BORDER_REFLECT) if ph or pw else gray
    x = torch.from_numpy(np.repeat(g[None, None].astype(np.float32) / 255.0, 3, 1)).to(dev)
    with torch.no_grad(), torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda" and SR_MODELS[key]["fp16"]):
        y = model(x)
    out = (torch.nan_to_num(y.float()).clamp(0, 1).mean(1)[0].cpu().numpy() * 255 + 0.5).astype(np.uint8)
    return out[: 2 * h, : 2 * w]


@router.get("/api/dz/image/{image_id}/ai/{model}/{xy}.jpg")
def image_ai_tile(image_id: int, model: str, xy: str, d: str = "preview"):
    """A tile of the AI-enhanced level (2x the full-resolution grid). Invented detail: display only."""
    if model not in SR_MODELS or not sr_available():
        raise HTTPException(404, "AI enhance is not available on this server")
    entry = _pyramid(image_id)
    out = TILE_DIR / f"img-v{IMG_TILE_VERSION}" / f"ai-{model}" / entry["kind"] / str(image_id) / f"{xy}.jpg"
    if not out.exists():
        try:
            x, y = (int(v) for v in xy.split("_"))
        except ValueError:
            raise HTTPException(400)
        full = entry["full"][:, :, 0]
        H, W = full.shape
        ox0, oy0 = x * TILE - (1 if x else 0), y * TILE - (1 if y else 0)
        ox1, oy1 = min(2 * W, (x + 1) * TILE + 1), min(2 * H, (y + 1) * TILE + 1)
        if ox0 >= 2 * W or oy0 >= 2 * H:
            raise HTTPException(404)
        ix0, iy0 = max(0, ox0 // 2 - SR_PAD), max(0, oy0 // 2 - SR_PAD)
        ix1, iy1 = min(W, -(-ox1 // 2) + SR_PAD), min(H, -(-oy1 // 2) + SR_PAD)
        crop = full[iy0:iy1, ix0:ix1]
        if not crop.any():                         # outside the footprint: nothing to invent
            up = np.zeros((2 * crop.shape[0], 2 * crop.shape[1]), np.uint8)
        else:
            with _sr_lock:                         # one GPU job at a time
                up = _super_resolve(crop, model)
        tile = up[oy0 - 2 * iy0: oy1 - 2 * iy0, ox0 - 2 * ix0: ox1 - 2 * ix0]
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), tile, [cv2.IMWRITE_JPEG_QUALITY, SCIENCE_Q])
    return FileResponse(out, headers={"Cache-Control": "public, max-age=604800"})


# ---------- the sky around an alert (Legacy Surveys) ----------

LEGACY = "https://www.legacysurvey.org/viewer/cutout.jpg"
LAYERS = ["ls-dr10", "ls-dr9", "sdss"]          # tried in order; the first with data wins
NATIVE = {"ls-dr10": 0.262, "ls-dr9": 0.262, "sdss": 0.396}
SKY_WIDTH = 8192          # virtual image (px); the deepest level is 2x oversampled by the survey's resampler
OVERSAMPLE = 2
_http = requests.Session()
_http.headers["User-Agent"] = "cosmic-watch-ai (github.com/jay4linkh/cosmic-watch-ai)"
_layer_cache: dict[str, str | None] = {}


def _fetch(ra, dec, pixscale, size, layer, timeout=40):
    r = _http.get(LEGACY, params={"ra": f"{ra:.6f}", "dec": f"{dec:.6f}", "layer": layer,
                                  "pixscale": f"{pixscale:.5f}", "size": size}, timeout=timeout)
    r.raise_for_status()
    img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("not an image")
    return r.content, img


def _covered(img):
    return float((img.max(2) > 8).mean())


def pick_layer(ra, dec):
    """The first survey layer that actually has data here (cached per ~1 arcmin cell)."""
    key = f"{ra:.2f},{dec:.2f}"
    if key not in _layer_cache:
        _layer_cache[key] = None
        for layer in LAYERS:
            try:
                _, img = _fetch(ra, dec, 2.0, 128, layer, timeout=25)
            except Exception as e:  # noqa: BLE001
                log.warning("legacy probe %s failed: %s", layer, e)
                continue
            if _covered(img) > 0.6:
                _layer_cache[key] = layer
                break
    return _layer_cache[key]


def _tan_inverse(ra0, dec0, xi, eta):
    """Gnomonic (TAN) deprojection; xi/eta in radians, east/north positive."""
    a0, d0 = math.radians(ra0), math.radians(dec0)
    rho = math.hypot(xi, eta)
    if rho == 0:
        return ra0, dec0
    c = math.atan(rho)
    dec = math.asin(math.cos(c) * math.sin(d0) + eta * math.sin(c) * math.cos(d0) / rho)
    ra = a0 + math.atan2(xi * math.sin(c), rho * math.cos(d0) * math.cos(c) - eta * math.sin(d0) * math.sin(c))
    return math.degrees(ra) % 360, math.degrees(dec)


def _tan_forward(ra0, dec0, ra, dec):
    a0, d0, a, d = map(math.radians, (ra0, dec0, ra, dec))
    cosc = math.sin(d0) * math.sin(d) + math.cos(d0) * math.cos(d) * math.cos(a - a0)
    xi = math.cos(d) * math.sin(a - a0) / cosc
    eta = (math.cos(d0) * math.sin(d) - math.sin(d0) * math.cos(d) * math.cos(a - a0)) / cosc
    return xi, eta


def _media_url(p):
    if not p:
        return None
    path = Path(p)
    path = path if path.is_absolute() else ROOT / path
    try:
        return "/media/" + path.resolve().relative_to(DATA.resolve()).as_posix()
    except ValueError:
        return None


def _sky_geometry(layer):
    max_level = _levels(SKY_WIDTH, SKY_WIDTH)
    deepest = NATIVE[layer] / OVERSAMPLE        # arcsec per virtual pixel at the deepest level
    return max_level, deepest


@router.get("/api/dz/sky")
def sky_info(ra: float = Query(ge=0, lt=360), dec: float = Query(ge=-90, le=90)):
    """Tile source for the sky around (ra, dec), plus every alert we know of inside it."""
    layer = pick_layer(ra, dec)
    if not layer:
        return {"available": False, "reason": "No deep survey imaging covers this part of the sky."}
    max_level, deepest = _sky_geometry(layer)
    half_deg = SKY_WIDTH / 2 * deepest / 3600
    con = db.connect()
    try:
        rows = con.execute("""SELECT survey, oid, cls, probability, ra, dec, stamp_path FROM alerts
                              WHERE dec BETWEEN ? AND ?""", (dec - half_deg, dec + half_deg)).fetchall()
    finally:
        con.close()
    marks = []
    for r in rows:
        xi, eta = _tan_forward(ra, dec, r["ra"], r["dec"])
        px = SKY_WIDTH / 2 - math.degrees(xi) * 3600 / deepest        # east is left
        py = SKY_WIDTH / 2 - math.degrees(eta) * 3600 / deepest       # north is up
        if 0 <= px < SKY_WIDTH and 0 <= py < SKY_WIDTH:
            marks.append({"survey": r["survey"], "oid": r["oid"], "cls": r["cls"], "p": r["probability"],
                          "x": px / SKY_WIDTH, "y": py / SKY_WIDTH,
                          "stamp_url": _media_url(r["stamp_path"])})
    return {"available": True, "layer": layer, "ra": ra, "dec": dec,
            "width": SKY_WIDTH, "height": SKY_WIDTH, "tileSize": TILE, "tileOverlap": 0, "maxLevel": max_level,
            "arcsec_per_px": deepest, "native_arcsec_per_px": NATIVE[layer], "oversample": OVERSAMPLE,
            "field_arcmin": round(SKY_WIDTH * deepest / 60, 1),
            "tiles": f"/api/dz/sky/{layer}/{ra:.6f}_{dec:.6f}/{{level}}/{{x}}_{{y}}.jpg",
            "alerts": marks,
            "credit": "Sky: DESI Legacy Imaging Surveys (legacysurvey.org)"}


@router.get("/api/dz/sky/{layer}/{pos}/{level}/{xy}.jpg")
def sky_tile(layer: str, pos: str, level: int, xy: str):
    if layer not in NATIVE:
        raise HTTPException(404)
    try:
        ra0, dec0 = (float(v) for v in pos.split("_"))
        x, y = (int(v) for v in xy.split("_"))
    except ValueError:
        raise HTTPException(400)
    out = TILE_DIR / "sky" / layer / pos / str(level) / f"{xy}.jpg"
    if not out.exists():
        max_level, deepest = _sky_geometry(layer)
        if level < 0 or level > max_level:
            raise HTTPException(404)
        width = SKY_WIDTH / 2 ** (max_level - level)
        if x * TILE >= width or y * TILE >= width:
            raise HTTPException(404)
        size = int(min(TILE, width - x * TILE))
        ps = deepest * 2 ** (max_level - level)                       # arcsec per pixel at this level
        cx, cy = x * TILE + size / 2, y * TILE + size / 2
        xi = -math.radians((cx - width / 2) * ps / 3600)
        eta = -math.radians((cy - width / 2) * ps / 3600)
        ra, dec = _tan_inverse(ra0, dec0, xi, eta)
        try:
            raw, _ = _fetch(ra, dec, ps, size, layer)
        except Exception as e:  # noqa: BLE001
            log.warning("legacy tile %s/%s/%s failed: %s", pos, level, xy, e)
            raise HTTPException(502, "survey tile unavailable")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(raw)
    return FileResponse(out, headers={"Cache-Control": "public, max-age=604800"})


# ---------- host galaxy ----------

HOST_SIZE = 256           # native-resolution cut-out (px) around the alert
HOST_MAX_SEP = 12.0       # arcsec: how far from the alert a host may be


@router.get("/api/dz/host")
def host_galaxy(ra: float = Query(ge=0, lt=360), dec: float = Query(ge=-90, le=90)):
    """Find the galaxy an alert sits in, and let our CNN classify it."""
    from .ml import infer as cnn
    cache = TILE_DIR / "host" / f"{ra:.6f}_{dec:.6f}.json"
    if cache.exists():
        out = json.loads(cache.read_text())
        if out.get("model") == (cnn.version() if cnn.available() else None):
            return out
    layer = pick_layer(ra, dec)
    if not layer:
        return {"found": False, "reason": "no survey coverage"}
    ps = NATIVE[layer]
    try:
        _, img = _fetch(ra, dec, ps, HOST_SIZE, layer)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"survey unavailable: {e}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    import sep
    data = gray.astype(np.float32)
    bkg = sep.Background(data, bw=32, bh=32)
    sub = data - bkg
    objs = sep.extract(sub, 3.0, err=bkg.globalrms, minarea=12, deblend_cont=0.01)
    c = HOST_SIZE / 2
    best = None
    for o in objs:
        d = math.hypot(o["x"] - c, o["y"] - c) * ps
        # extended sources only; prefer close and big (a large galaxy can be offset from the event)
        if o["a"] < 2.2 or d > HOST_MAX_SEP + 2 * o["a"] * ps:
            continue
        score = d / max(o["a"] * ps, 0.5)
        if best is None or score < best[0]:
            best = (score, o, d)
    if best is None:
        out = {"found": False, "reason": "no extended galaxy within 12 arcsec (hostless, or too faint for the survey)"}
    else:
        _, o, d = best
        result = {"found": True, "layer": layer, "sep_arcsec": round(d, 2),
                  "size_arcsec": round(float(o["a"]) * ps * 2, 1),
                  "ellip": round(1 - float(o["b"]) / max(float(o["a"]), 1e-6), 2),
                  # host position relative to the alert, in arcsec (east, north positive)
                  "dx_arcsec": round(-(float(o["x"]) - c) * ps, 2), "dy_arcsec": round(-(float(o["y"]) - c) * ps, 2)}
        if cnn.available():
            probs = cnn.classify_cutouts(gray, [(float(o["x"]), float(o["y"]), float(o["a"]))])[0]
            order = np.argsort(-probs)
            result.update({"label": CLASSES[int(order[0])], "conf": round(float(probs[order[0]]), 3),
                           "top": [[CLASSES[int(i)], round(float(probs[i]), 3)] for i in order[:4]],
                           "coarse": {k: round(float(probs[idx].sum()), 3) for k, idx in COARSE.items()}})
        out = result
    out["model"] = cnn.version() if cnn.available() else None
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out))
    return out
