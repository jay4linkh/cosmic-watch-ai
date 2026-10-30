import os
from pathlib import Path


def _env_int(name, default):
    return int(os.environ.get(name, default))


ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("COSMIC_DATA_DIR", ROOT / "data"))
DB_PATH = DATA / "cosmic_watch.db"
IMG_DIR = DATA / "images"
STAMP_DIR = DATA / "stamps"

# How often each source is polled by the pipeline loop (seconds).
INTERVALS = {
    "sdo": 15 * 60,      # SDO publishes new solar images every ~15 min
    "alerts": 10 * 60,   # Rubin + ZTF alerts via ALeRCE arrive within minutes
    "mast": 6 * 3600,    # JWST / Hubble public releases land a few times a day
}

# MAST: how far back to look for newly public images, and how many to process per run.
MAST_LOOKBACK_DAYS = 3
MAST_MAX_PER_RUN = _env_int("MAST_MAX_PER_RUN", 10)

# Alerts: how many objects to pull per run, and how many get image stamps.
RUBIN_PAGE_SIZE = 200
# Stamps fetched per Rubin class per run (other classes: 2). Bogus kept small, for the real-vs-bogus demo.
RUBIN_STAMPS_PER_CLASS = {"AGN": 10, "SN": 10, "TDE": 10, "VS": 6, "asteroid": 4, "satellite": 3, "bogus": 3}
ZTF_PER_CLASS = 8
ZTF_MAX_AGE_DAYS = 7

# Detector: downscale huge images to this longest side before source extraction.
DETECT_MAX_SIDE = _env_int("DETECT_MAX_SIDE", 3000)
