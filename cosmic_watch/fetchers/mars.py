"""Mars, daily: the newest raw images from NASA's Perseverance and Curiosity rovers (public, no key).

Raw images are published within hours of reaching Earth. We take the newest sol from each rover,
keep a varied set of full-size frames from the science and navigation cameras, and store them with
NASA's caption, camera and pointing.
"""
import logging
from datetime import datetime

import requests

from ..config import IMG_DIR

log = logging.getLogger("cosmic_watch")
_http = requests.Session()
_http.headers["User-Agent"] = "cosmic-watch-ai (github.com/jay4linkh/cosmic-watch-ai)"

PERSEVERANCE = "https://mars.nasa.gov/rss/api/"
CURIOSITY = "https://mars.nasa.gov/api/v1/raw_image_items/"
PER_CAMERA = 3
PER_ROVER = 14

ROVERS = {
    "perseverance": {"name": "Perseverance", "site": "Jezero crater", "lat": 18.44, "lon": 77.45},
    "curiosity": {"name": "Curiosity", "site": "Gale crater (Mount Sharp)", "lat": -4.59, "lon": 137.44},
}
# camera id -> (friendly name, what it is for); others are skipped (engineering / thumbnails)
CAMERAS = {
    "MCZ_LEFT": ("Mastcam-Z left", "zoomable colour science camera"),
    "MCZ_RIGHT": ("Mastcam-Z right", "zoomable colour science camera"),
    "NAVCAM_LEFT": ("Navcam left", "navigation camera, wide view of the terrain"),
    "NAVCAM_RIGHT": ("Navcam right", "navigation camera, wide view of the terrain"),
    "FRONT_HAZCAM_LEFT_A": ("Front Hazcam", "hazard camera near the wheels"),
    "SHERLOC_WATSON": ("SHERLOC WATSON", "close-up camera on the robotic arm (rock textures)"),
    "SUPERCAM_RMI": ("SuperCam RMI", "telescopic camera aimed at laser targets"),
    "MAST_LEFT": ("Mastcam left", "colour science camera"),
    "MAST_RIGHT": ("Mastcam right", "telephoto colour science camera"),
    "NAV_LEFT_B": ("Navcam left", "navigation camera"),
    "NAV_RIGHT_B": ("Navcam right", "navigation camera"),
    "FHAZ_LEFT_B": ("Front Hazcam", "hazard camera near the wheels"),
    "MAHLI": ("MAHLI", "hand-lens close-up camera on the arm"),
    "CHEMCAM_RMI": ("ChemCam RMI", "telescopic camera aimed at laser targets"),
}


def _perseverance():
    j = _http.get(PERSEVERANCE, params={"feed": "raw_images", "category": "mars2020", "feedtype": "json",
                                         "num": 200, "page": 0, "order": "sol desc"}, timeout=60).json()
    out = []
    for im in j.get("images", []):
        cam = im["camera"]["instrument"]
        if cam not in CAMERAS or im.get("sample_type") == "Thumbnail":
            continue
        files = im.get("image_files", {})
        out.append({"rover": "perseverance", "id": im["imageid"], "sol": im["sol"], "camera": cam,
                    "url": files.get("large") or files.get("medium") or files.get("full_res"),
                    "taken": im.get("date_taken_utc"), "caption": (im.get("caption") or "").strip(),
                    "az": float(im["extended"].get("mastAz") or 0), "el": float(im["extended"].get("mastEl") or 0)})
    return out


def _curiosity():
    j = _http.get(CURIOSITY, params={"order": "sol desc,date_taken desc", "per_page": 200, "page": 0,
                                      "condition_1": "msl:mission"}, timeout=60).json()
    out = []
    for im in j.get("items", []):
        cam = im.get("instrument")
        if cam not in CAMERAS or im.get("is_thumbnail") or (im.get("extended") or {}).get("sample_type") != "full":
            continue
        ext = im.get("extended") or {}
        out.append({"rover": "curiosity", "id": im["imageid"], "sol": im["sol"], "camera": cam, "url": im["url"],
                    "taken": im.get("date_taken"), "caption": (im.get("description") or "").strip(),
                    "az": float(ext.get("mast_az") or 0), "el": float(ext.get("mast_el") or 0)})
    return out


def _pick(items):
    """Newest sol first, at most PER_CAMERA frames per camera, PER_ROVER in total."""
    items.sort(key=lambda r: (-r["sol"], r["taken"] or ""), reverse=False)
    per_cam, out = {}, []
    for r in items:
        if per_cam.get(r["camera"], 0) >= PER_CAMERA:
            continue
        per_cam[r["camera"]] = per_cam.get(r["camera"], 0) + 1
        out.append(r)
        if len(out) >= PER_ROVER:
            break
    return out


def fetch(is_seen):
    """Yield image records (downloaded) for frames we have not stored yet."""
    for rover, fn in (("perseverance", _perseverance), ("curiosity", _curiosity)):
        try:
            items = _pick(fn())
        except Exception as e:  # noqa: BLE001
            log.warning("Mars %s feed failed: %s", rover, e)
            continue
        for r in items:
            if is_seen("mars", r["id"]) or not r["url"]:
                continue
            d = IMG_DIR / "mars" / rover
            d.mkdir(parents=True, exist_ok=True)
            ext = ".png" if r["url"].lower().endswith(".png") else ".jpg"
            path = d / f"{r['id']}{ext}"
            try:
                if not path.exists():
                    resp = _http.get(r["url"], timeout=90)
                    resp.raise_for_status()
                    path.write_bytes(resp.content)
            except Exception as e:  # noqa: BLE001
                log.warning("Mars image %s: %s", r["id"], e)
                continue
            info = ROVERS[rover]
            cam_name, cam_desc = CAMERAS[r["camera"]]
            taken = r["taken"]
            try:
                obs = datetime.fromisoformat(taken.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M:%S") if taken else None
            except ValueError:
                obs = None
            yield {"source": "mars", "external_id": r["id"], "path": path,
                   "title": f"{info['name']} · {cam_name} · Sol {r['sol']}", "target": info["site"],
                   "instrument": cam_name, "filter": rover, "obs_time": obs,
                   "meta": {"rover": rover, "rover_name": info["name"], "site": info["site"], "sol": r["sol"],
                            "camera": r["camera"], "camera_name": cam_name, "camera_desc": cam_desc,
                            "caption": r["caption"][:600], "az": r["az"], "el": r["el"], "url": r["url"],
                            "lat": info["lat"], "lon": info["lon"]}}
