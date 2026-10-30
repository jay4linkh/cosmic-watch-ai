"""Live transient alerts from Rubin (LSST) and ZTF, via the ALeRCE broker."""
import warnings

import numpy as np
import requests
from astropy.time import Time
from alerce.core import Alerce

from ..config import RUBIN_PAGE_SIZE, ZTF_MAX_AGE_DAYS, ZTF_PER_CLASS

warnings.filterwarnings("ignore", module="alerce")

ZTF_API = "https://api.alerce.online/ztf/v1/objects/"

# Classes that point to a (supermassive) black hole at work.
BH_CLASSES = {"TDE", "AGN", "QSO", "Blazar"}

# (classifier, class) pairs we pull from ZTF every run.
ZTF_QUERIES = [
    ("LC_classifier_ATAT_forced_phot(beta)", "TDE"),   # star torn apart by a black hole
    ("lc_classifier", "AGN"),
    ("lc_classifier", "QSO"),
    ("lc_classifier", "Blazar"),
    ("stamp_classifier", "SN"),
]

_client = None


def client() -> Alerce:
    global _client
    if _client is None:
        _client = Alerce()
    return _client


def rubin_latest(page_size: int = RUBIN_PAGE_SIZE) -> list[dict]:
    """Most recently updated Rubin objects with their top stamp-classifier class."""
    df = client().query_objects(survey="lsst", page_size=page_size, order_by="lastmjd", order_mode="DESC")
    if df is None or len(df) == 0:
        return []
    df = df[df["ranking"] == 1]
    # Several classifier versions can score the same object; keep the newest version.
    df = df.sort_values(["classifier_name", "classifier_version"], ascending=False).drop_duplicates("oid")
    out = []
    for r in df.itertuples():
        out.append({
            "survey": "lsst", "oid": str(r.oid), "cls": str(r.class_name),
            "classifier": str(r.classifier_name), "probability": float(r.probability),
            "lastmjd": float(r.lastmjd), "ra": float(r.meanra), "dec": float(r.meandec),
            "n_det": int(r.n_det), "bh_candidate": int(r.class_name in BH_CLASSES),
        })
    return sorted(out, key=lambda a: a["lastmjd"], reverse=True)


def ztf_candidates(per_class: int = ZTF_PER_CLASS, max_age_days: float = ZTF_MAX_AGE_DAYS) -> list[dict]:
    """Recent ZTF objects classified as black-hole activity (TDE/AGN/QSO/Blazar) or supernovae."""
    min_mjd = Time.now().mjd - max_age_days
    out = []
    for classifier, cls in ZTF_QUERIES:
        r = requests.get(ZTF_API, params={
            "classifier": classifier, "class": cls, "page_size": per_class,
            "order_by": "lastmjd", "order_mode": "DESC",
        }, timeout=90)
        r.raise_for_status()
        seen = set()
        for it in r.json().get("items", []):
            if it["oid"] in seen or it["lastmjd"] < min_mjd:
                continue
            seen.add(it["oid"])
            out.append({
                "survey": "ztf", "oid": it["oid"], "cls": cls, "classifier": classifier,
                "probability": float(it["probability"]), "lastmjd": float(it["lastmjd"]),
                "ra": float(it["meanra"]), "dec": float(it["meandec"]), "n_det": int(it["ndet"]),
                "bh_candidate": int(cls in BH_CLASSES),
            })
    return out


def stamps(survey: str, oid: str) -> dict[str, np.ndarray] | None:
    """Science / template / difference cutouts as float arrays."""
    oid_arg = int(oid) if survey == "lsst" else oid
    st = client().get_stamps(oid_arg, survey=survey)
    if st is None:
        return None
    if isinstance(st, dict):          # Rubin: {'cutoutScience': HDU, ...}
        hdus = [st.get("cutoutScience"), st.get("cutoutTemplate"), st.get("cutoutDifference")]
    else:                             # ZTF: HDUList [science, template, difference]
        hdus = list(st)[:3]
    if any(h is None or h.data is None for h in hdus):
        return None
    sci, tmpl, diff = (np.nan_to_num(np.asarray(h.data, dtype=np.float32)) for h in hdus)
    return {"science": sci, "template": tmpl, "difference": diff}
