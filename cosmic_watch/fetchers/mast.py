"""Newly public JWST and Hubble images from the MAST archive."""
import logging
import time

import requests
from astropy.time import Time
from astroquery.mast import Observations

from ..config import IMG_DIR, MAST_LOOKBACK_DAYS, MAST_MAX_PER_RUN

DOWNLOAD = "https://mast.stsci.edu/api/v0.1/Download/file"
DOWNLOAD_DEADLINE_S = 180

log = logging.getLogger("cosmic_watch")


def _score(row) -> int:
    """Prefer combined/detection images (deepest, best for finding galaxies) over single exposures."""
    f = str(row["filters"]).lower()
    oid = str(row["obs_id"]).lower()
    combined = oid.startswith(("hst_", "jw"))
    return (2 if f in ("detection", "total") else 1) + (2 if combined else 0)


def find_new(days: float = MAST_LOOKBACK_DAYS):
    """Return recently released science images, one per (mission, target, instrument), newest first."""
    now = Time.now().mjd
    picked = {}
    for coll in ("JWST", "HST"):
        obs = Observations.query_criteria(
            obs_collection=coll, dataproduct_type="image", calib_level=[3],
            intentType="science", t_obs_release=[now - days, now],
        )
        for row in obs:
            if not str(row["jpegURL"]).startswith("mast:"):
                continue
            key = (coll, str(row["target_name"]), str(row["instrument_name"]))
            best = picked.get(key)
            if best is None or (_score(row), row["t_obs_release"]) > (_score(best), best["t_obs_release"]):
                picked[key] = row
    rows = sorted(picked.values(), key=lambda r: r["t_obs_release"], reverse=True)
    return rows


def _get(uri: str, path, deadline_s: float = DOWNLOAD_DEADLINE_S):
    """Stream a file with a total time limit (requests' timeout is per read, not per download)."""
    start = time.monotonic()
    tmp = path.with_suffix(".part")
    with requests.get(DOWNLOAD, params={"uri": uri}, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 16):
                f.write(chunk)
                if time.monotonic() - start > deadline_s:
                    raise TimeoutError(f"download exceeded {deadline_s:.0f}s: {uri}")
    tmp.replace(path)


def download(row):
    coll = str(row["obs_collection"]).lower()
    out_dir = IMG_DIR / coll
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{row['obs_id']}.jpg"
    if not path.exists():
        try:
            _get(str(row["jpegURL"]), path)
        finally:
            path.with_suffix(".part").unlink(missing_ok=True)
    return {
        "source": coll,
        "external_id": str(row["obs_id"]),
        "title": f"{row['obs_collection']} {row['instrument_name']}: {row['target_name']}",
        "target": str(row["target_name"]),
        "instrument": str(row["instrument_name"]),
        "filter": str(row["filters"]),
        "obs_time": Time(row["t_obs_release"], format="mjd").strftime("%Y-%m-%d %H:%M:%S"),
        "path": path,
    }


def fetch(is_seen, limit: int = MAST_MAX_PER_RUN):
    """Yield downloaded records for images not yet processed (`is_seen(source, obs_id)` -> bool)."""
    n = 0
    for row in find_new():
        if n >= limit:
            break
        if is_seen(str(row["obs_collection"]).lower(), str(row["obs_id"])):
            continue
        try:
            rec = download(row)
        except Exception as e:      # skip one slow/broken file, keep going
            log.warning("MAST download skipped %s: %s", row["obs_id"], e)
            continue
        yield rec
        n += 1
