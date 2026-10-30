"""GCN Circulars: astronomers' live reports on cosmic explosions (NASA's General Coordinates Network).

Every gamma-ray burst (Swift, Fermi, SVOM), Einstein Probe X-ray flash, IceCube neutrino, gravitational-wave
follow-up and fast optical transient gets circulars within minutes to hours. We group the newest ones by
event, classify the event type from its name, and pull out sky coordinates and redshift when a report gives them.
"""
import logging
import re
from datetime import datetime, timezone

import requests

log = logging.getLogger("cosmic_watch")
BASE = "https://gcn.nasa.gov/circulars"
_http = requests.Session()
_http.headers["User-Agent"] = "cosmic-watch-ai (github.com/jay4linkh/cosmic-watch-ai)"

KINDS = [
    (r"^GRB\s?\d{6}", "grb", "Gamma-ray burst"),
    (r"^EP\s?\d{6}|^EP-WXT|^EP Alert|Einstein Probe", "xray", "X-ray flash (Einstein Probe)"),
    (r"^IceCube|^ICECUBE|neutrino", "neutrino", "High-energy neutrino"),
    (r"^S\d{6}[a-z]+|^LIGO|^GW\d", "gw", "Gravitational-wave follow-up"),
    (r"^FRB", "frb", "Fast radio burst"),
    (r"^(AT|SN)\s?20\d\d", "optical", "Optical transient"),
    (r"^ZTF\d|^ATLAS|^Swift J|^MAXI", "transient", "Transient"),
]
EXPLAIN = {
    "grb": "A gamma-ray burst: the most powerful explosions known. Long ones (more than ~2 s) come from collapsing massive "
           "stars forming black holes; short ones from merging neutron stars. Satellites catch the flash, then telescopes "
           "race to see the fading afterglow and measure its distance (redshift).",
    "xray": "An X-ray flash caught by China's Einstein Probe wide-field telescope: often a gamma-ray-burst cousin, a stellar "
            "flare, or a newborn black hole or magnetar.",
    "neutrino": "IceCube, a cubic kilometre of Antarctic ice, caught a high-energy neutrino. Its direction is searched for "
                "a source: blazars and tidal disruption events are suspects.",
    "gw": "Follow-up of a gravitational-wave alert: telescopes search the sky map for light from the collision.",
    "frb": "A fast radio burst: a millisecond radio flash from another galaxy, probably from a magnetar.",
    "optical": "A new optical transient: a supernova, a tidal disruption event or another fast-changing source.",
    "transient": "A transient source reported by a survey or monitor.",
}
NAME = re.compile(r"(GRB\s?\d{6}[A-Z]?|EP\d{6}[a-z]|EP-WXT trigger \d+|EP Alert [\d.]+|IceCube-\d{6}[A-Z]?|"
                  r"S\d{6}[a-z]+|FRB\s?\d{8}[A-Z]?|AT\s?20\d\d[a-z]+|SN\s?20\d\d[a-z]+)")
COORD = re.compile(r"RA,?\s*Dec(?:\s*\(J2000\))?\s*=\s*([0-9]{1,3}\.[0-9]+)\s*,?\s*([-+]?[0-9]{1,2}\.[0-9]+)", re.I)
REDSHIFT = re.compile(r"\bz\s*[=~]\s*([0-9]\.[0-9]{1,4})", re.I)


def kind_of(name, subject):
    for pat, k, label in KINDS:
        if re.search(pat, name or "") or re.search(pat, subject or ""):
            return k, label
    return "other", "Other"


def _index(pages=1):
    out = []
    for p in range(1, pages + 1):
        r = _http.get(BASE, params={"page": p} if p > 1 else None, timeout=60)
        r.raise_for_status()
        for cid, subj in re.findall(r'href="/circulars/(\d+)">([^<]+)</a>', r.text):
            out.append((int(cid), subj.strip()))
    return out


def fetch(known: dict | None = None, n=60):
    """Newest circulars grouped by event. `known` maps circular id -> parsed circular (skip re-downloads)."""
    known = known or {}
    circ = {}
    for cid, subj in _index()[:n]:
        if str(cid) in known:
            circ[cid] = known[str(cid)]
            continue
        try:
            j = _http.get(f"{BASE}/{cid}.json", timeout=30).json()
        except Exception as e:  # noqa: BLE001
            log.warning("GCN %s: %s", cid, e)
            continue
        body = j.get("body") or ""
        m, z = COORD.search(body), REDSHIFT.search(body)
        subject = j.get("subject", subj)
        name = NAME.search(j.get("eventId") or "") or NAME.search(subject)
        created = j.get("createdOn")
        circ[cid] = {"id": cid, "subject": subject,
                     "event": name.group(1) if name else (j.get("eventId") or subject.split(":")[0].strip()),
                     "submitter": (j.get("submitter") or "").split("<")[0].strip(),
                     "created": datetime.fromtimestamp(created / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if created else None,
                     "ra": float(m.group(1)) if m else None, "dec": float(m.group(2)) if m else None,
                     "redshift": float(z.group(1)) if z else None,
                     "snippet": re.sub(r"\s+", " ", body)[:600]}
    events = {}
    for c in sorted(circ.values(), key=lambda c: c["id"]):
        ev = events.setdefault(c["event"], {"event": c["event"], "circulars": [], "ra": None, "dec": None, "redshift": None})
        ev["circulars"].append({k: c[k] for k in ("id", "subject", "submitter", "created")})
        for k in ("ra", "dec", "redshift"):
            ev[k] = ev[k] if ev[k] is not None else c[k]
    out = []
    for ev in events.values():
        k, label = kind_of(ev["event"], ev["circulars"][0]["subject"])
        ev.update(kind=k, kind_label=label, explain=EXPLAIN.get(k), latest=ev["circulars"][-1]["created"],
                  first=ev["circulars"][0]["created"], n=len(ev["circulars"]),
                  url=f"{BASE}/{ev['circulars'][-1]['id']}")
        out.append(ev)
    out.sort(key=lambda e: e["latest"] or 0, reverse=True)
    return {"events": out, "circulars": {str(k): v for k, v in circ.items()}}
