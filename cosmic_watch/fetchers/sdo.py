"""Latest full-disk Sun images from NASA's Solar Dynamics Observatory (~15 min cadence)."""
import hashlib
from email.utils import parsedate_to_datetime

import requests

from ..config import IMG_DIR

URL = "https://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_{code}.jpg"

# code -> (title, what the detector looks for)
CHANNELS = {
    "HMIIC": ("SDO HMI continuum (visible light)", "sunspots"),
    "0171": ("SDO AIA 171 Å (quiet corona, loops)", "active_regions"),
    "0193": ("SDO AIA 193 Å (corona, coronal holes)", "active_regions"),
    "0304": ("SDO AIA 304 Å (chromosphere, prominences)", "active_regions"),
    "0131": ("SDO AIA 131 Å (flaring plasma, ~10 million °C)", "active_regions"),
    "0094": ("SDO AIA 94 Å (hot flare loops, ~6 million °C)", "active_regions"),
}


def fetch():
    """Yield one record per channel with the downloaded image path."""
    out_dir = IMG_DIR / "sdo"
    out_dir.mkdir(parents=True, exist_ok=True)
    for code, (title, mode) in CHANNELS.items():
        r = requests.get(URL.format(code=code), timeout=60)
        r.raise_for_status()
        digest = hashlib.sha1(r.content).hexdigest()[:12]
        lm = r.headers.get("Last-Modified")
        obs = parsedate_to_datetime(lm).strftime("%Y-%m-%d %H:%M:%S") if lm else None
        stamp = obs.replace("-", "").replace(":", "").replace(" ", "_") if obs else digest
        path = out_dir / f"{code}_{stamp}.jpg"
        if not path.exists():
            path.write_bytes(r.content)
        yield {
            "source": "sdo",
            "external_id": f"{code}:{digest}",
            "title": title,
            "target": "Sun",
            "instrument": "AIA" if code.isdigit() else "HMI",
            "filter": code,
            "obs_time": obs,
            "path": path,
            "mode": mode,
        }
