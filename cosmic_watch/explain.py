"""The Cosmic Agent explains what you clicked: "what is this, really?"

For one detection it gathers measured facts (size in arcseconds, how concentrated its light is,
the filter's wavelength and what that filter isolates, what the field is, the CNN's call and
how sure it is), weighs the evidence for "galaxy" against the usual impostors (glowing nebula
gas, stars, satellite trails), and adds background knowledge for the verdict.

The CNN only knows 10 galaxy shapes, so it will call a filament of gas "Disturbed": the verdict
here is how the agent catches that.

If Claude credentials are available (ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN or an `ant auth login`
profile) and the `anthropic` package is installed, Claude also writes a short narrative from the
same facts. Without them everything still works from the built-in knowledge.
"""
import json
import logging
import math
import os
import re
from pathlib import Path

import cv2
import numpy as np

from .config import DATA, ROOT
from .ml.classes import CLASSES, COARSE

log = logging.getLogger("cosmic_watch")
CACHE_DIR = DATA / "explain"
CLAUDE_MODEL = "claude-opus-5-5"

# arcsec per pixel of MAST's drizzled products (approximate; previews keep the product's grid)
PIXEL_SCALE = {
    "ACS/WFC": 0.05, "WFC3/UVIS": 0.04, "WFC3/IR": 0.128,
    "NIRISS/IMAGE": 0.0656, "MIRI/IMAGE": 0.11,
}

# narrow-band filters isolate one emission line: what glows there
LINES = {
    "F656N": "hydrogen-alpha (Hα, 656 nm): glowing ionised hydrogen",
    "F658N": "hydrogen-alpha + nitrogen [N II] (658 nm): glowing ionised gas",
    "F502N": "doubly ionised oxygen [O III] (502 nm): hot, ionised gas",
    "F487N": "hydrogen-beta (Hβ, 487 nm): ionised hydrogen",
    "F673N": "ionised sulphur [S II] (673 nm): shocked gas",
    "F187N": "Paschen-alpha (1.87 µm): ionised hydrogen, seen through dust",
    "F164N": "iron [Fe II] (1.64 µm): shocks from jets and outflows",
    "F212N": "molecular hydrogen H₂ (2.12 µm): warm gas at cloud edges",
    "F405N": "Brackett-alpha (4.05 µm): ionised hydrogen deep in dust",
    "F470N": "molecular hydrogen H₂ (4.70 µm)",
    "F335M": "PAH molecules (3.3 µm): carbon-rich dust lit by young stars",
    "F770W": "PAH molecules (7.7 µm): carbon-rich dust",
    "F1130W": "PAH molecules (11.3 µm): carbon-rich dust",
}

# fields we know (target name pattern -> what it is)
KNOWN_TARGETS = [
    (r"N(GC)?-?3372|CARINA", "nebula", "the Carina Nebula (NGC 3372), a giant star-forming cloud about 7,500 light-years away in our Milky Way"),
    (r"N(GC)?-?1976|ORION|M42", "nebula", "the Orion Nebula, the nearest big star-forming region, about 1,350 light-years away"),
    (r"N(GC)?-?6611|M16|EAGLE", "nebula", "the Eagle Nebula (M16), home of the Pillars of Creation"),
    (r"N(GC)?-?2070|TARANTULA|30-?DOR", "nebula", "the Tarantula Nebula in the Large Magellanic Cloud"),
    (r"PILLAR|NEBULA|STAR-?FORMING|HII|H-II|CLOUD", "nebula", "a star-forming region: clouds of gas and dust where new stars are born"),
    (r"HUBBLE-ULTRA-DEEP|HUDF|UDF", "deep", "the Hubble Ultra Deep Field, a tiny patch of sky with ~10,000 galaxies, some seen as they were 13 billion years ago"),
    (r"A2744|ABELL-?2744|PANDORA", "cluster", "Abell 2744 (Pandora's Cluster), a merging galaxy cluster whose gravity lenses even more distant galaxies"),
    (r"PARALLEL|DEEP|COSMOS|GOODS|CEERS", "deep", "a deep extragalactic field: almost every fuzzy object is a distant galaxy"),
    (r"N(GC)?-?121\b", "cluster", "NGC 121, one of the oldest star clusters in the Small Magellanic Cloud: most sources are stars"),
    (r"N(GC)?-?1792", "galaxy", "NGC 1792, a dusty star-forming spiral galaxy about 40 million light-years away"),
    (r"SN-?2008S", "galaxy", "SN 2008S, a 'supernova impostor' in the galaxy NGC 6946: a dust-shrouded star that flared up"),
    (r"^SN|SUPERNOVA", "galaxy", "the host galaxy of a recent supernova"),
]

KNOWLEDGE = {
    "nebula": ("Glowing gas, not a galaxy",
               "Nebulae are clouds of gas and dust inside our own galaxy. Ultraviolet light from hot young stars ionises the gas, "
               "and it glows in emission lines; dense, dusty parts are carved into pillars, ridges and filaments by that light and "
               "by stellar winds. These shapes are light-years long but only thousands of light-years away, so they look big and "
               "irregular. Our CNN only knows galaxy shapes, so it usually calls them 'Disturbed'."),
    "star": ("A star in our own galaxy",
             "Stars are point sources: their light comes from a single point blurred only by the telescope's optics (the point "
             "spread function). JWST's hexagonal mirror gives bright stars their six-pointed spikes; Hubble's give four."),
    "artifact": ("Probably an artifact",
                 "Very long, thin, straight features are usually satellite trails, diffraction spikes from a bright star, or "
                 "detector edges, not astronomical objects."),
    "merger": ("Galaxies colliding",
               "When galaxies pass close, gravity pulls out tidal tails and bridges of stars; over hundreds of millions of years "
               "they merge into one. Mergers trigger bursts of star formation and feed central black holes."),
    "elliptical": ("A smooth, round galaxy",
                   "Ellipticals are balls of mostly old, red stars with little gas left to form new ones. Many are the end products "
                   "of galaxy mergers; the biggest sit at the centres of galaxy clusters."),
    "spiral": ("A spiral galaxy",
               "Spirals are rotating disks where density waves bunch up gas into arms that light up with young blue stars. Our Milky "
               "Way is a barred spiral: about two thirds of spirals have a central bar of stars."),
    "edge-on": ("A disk galaxy seen edge-on",
                "This is a flat spinning disk tilted so we see it from the side; a dust lane often runs along the middle and a "
                "bulge may poke out above and below."),
}
CLASS_GROUP = {c: g for g, idx in COARSE.items() for c in (CLASSES[i] for i in idx)}
CLASS_NOTE = {
    "Disturbed": "irregular light with no clear shape: a galaxy disrupted by a neighbour (or, outside deep fields, often not a galaxy at all)",
    "Merging": "two or more galaxies interacting or merging",
    "Round smooth": "a round, smooth elliptical galaxy",
    "In-between round smooth": "a slightly flattened smooth galaxy",
    "Cigar-shaped smooth": "a very elongated smooth galaxy",
    "Barred spiral": "a spiral with a central bar of stars",
    "Unbarred tight spiral": "a spiral with tightly wound arms",
    "Unbarred loose spiral": "a spiral with open, loosely wound arms",
    "Edge-on without bulge": "a thin disk seen edge-on",
    "Edge-on with bulge": "a disk seen edge-on with a central bulge",
}


# ---------- facts ----------

def _scale_arcsec(instrument, filt):
    if instrument == "NIRCAM/IMAGE":
        m = re.search(r"F(\d{3,4})", filt or "")
        return 0.031 if m and int(m.group(1)) < 240 else 0.063     # short- vs long-wavelength channel
    return PIXEL_SCALE.get(instrument)


def _wavelength(source, filt):
    """('2.77 µm', 'near-infrared') from a filter name like F277W."""
    m = re.search(r"F(\d{3,4})[A-Z]*", filt or "")
    if not m:
        return None, None
    n = int(m.group(1))
    um = n / 100 if source == "jwst" else (n * 10 if n < 200 else n) / 1000
    band = ("ultraviolet" if um < 0.4 else "visible" if um < 0.7 else "near-infrared" if um < 5 else "mid-infrared")
    return (f"{um:.2f} µm" if um >= 1 else f"{um * 1000:.0f} nm"), band


BAND_NOTE = {
    "ultraviolet": "UV light traces hot, young, massive stars",
    "visible": "visible light, roughly what our eyes would see",
    "near-infrared": "near-infrared sees through dust and shows old stars and very distant, redshifted galaxies",
    "mid-infrared": "mid-infrared shows warm dust and molecules rather than starlight",
}


def _field(target):
    t = (target or "").upper()
    for pat, kind, text in KNOWN_TARGETS:
        if re.search(pat, t):
            return kind, text
    return None, None


def _concentration(gray, x, y, a, b, theta):
    """Light in the inner third vs the outer ring of the isophote (about 3a): galaxies are centrally
    peaked (typically 4-15); patches of nebula gas are flat (about 1-3)."""
    H, W = gray.shape
    A, B = 3 * a, 3 * b
    r = int(2.2 * A + 4)
    x0, x1, y0, y1 = max(0, int(x) - r), min(W, int(x) + r + 1), max(0, int(y) - r), min(H, int(y) + r + 1)
    sub = gray[y0:y1, x0:x1].astype(np.float32)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dx, dy = xx - x, yy - y
    ct, st = math.cos(theta), math.sin(theta)
    rr = np.hypot((dx * ct + dy * st) / A, (-dx * st + dy * ct) / B)
    ring = sub[(rr > 1.3) & (rr < 2)]
    bg = float(np.median(ring)) if ring.size > 10 else float(np.median(sub))
    inner, outer = sub[rr < 0.33], sub[(rr > 0.66) & (rr < 1)]
    if inner.size < 3 or outer.size < 3:
        return None
    return float((inner.mean() - bg) / max(outer.mean() - bg, 1.0))


def facts_for(con, det_id):
    d = con.execute("""SELECT d.*, i.source, i.instrument, i.filter, i.target, i.title, i.path, i.stats, i.id AS image_id
                       FROM detections d JOIN images i ON i.id = d.image_id WHERE d.id=?""", (det_id,)).fetchone()
    if not d:
        return None
    stats = json.loads(d["stats"] or "{}")
    sc = stats.get("scale", 1.0) or 1.0
    x, y, a, b = d["x"] / sc, d["y"] / sc, max(d["a"] / sc, 0.5), max(d["b"] / sc, 0.5)
    sci = DATA / "science" / f"{d['image_id']}.png"
    p = sci if sci.exists() else (Path(d["path"]) if Path(d["path"]).is_absolute() else ROOT / d["path"])
    gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    conc = _concentration(gray, x, y, a, b, d["theta"] or 0.0) if gray is not None else None
    arcsec = _scale_arcsec(d["instrument"], d["filter"])
    wl, band = _wavelength(d["source"], d["filter"])
    filt = (d["filter"] or "").split(";")[-1].strip().upper()
    field_kind, field_text = _field(d["target"])
    probs = json.loads(d["cnn_probs"]) if d["cnn_probs"] else None
    top = sorted(((CLASSES[i], p) for i, p in enumerate(probs)), key=lambda t: -t[1])[:3] if probs else []
    return {
        "image": {"id": d["image_id"], "telescope": "JWST" if d["source"] == "jwst" else "Hubble", "instrument": d["instrument"],
                  "filter": d["filter"], "wavelength": wl, "band": band, "band_note": BAND_NOTE.get(band),
                  "emission_line": LINES.get(filt), "target": d["target"], "title": d["title"],
                  "field": field_text, "field_kind": field_kind, "data": "science FITS" if sci.exists() else "preview JPEG"},
        "detection": {"id": det_id, "label": d["label"], "flux": round(d["score"] or 0, 1),
                      "size_px": round(6 * a, 1),                                    # ~isophotal diameter
                      "size_arcsec": round(6 * a * arcsec, 2) if arcsec else None,
                      "ellipticity": round(1 - b / a, 2), "angle_deg": round(math.degrees(d["theta"] or 0)),
                      "concentration": round(conc, 2) if conc is not None else None},
        "cnn": {"label": d["cnn_label"], "conf": round(d["cnn_conf"], 3) if d["cnn_conf"] else None,
                "top": [[k, round(v, 3)] for k, v in top]} if d["cnn_label"] else None,
    }


# ---------- verdict ----------

def verdict(f):
    """Weigh the evidence: galaxy, or one of the usual impostors?"""
    det, img, cnn = f["detection"], f["image"], f["cnn"]
    why, gas = [], 0
    if det["label"] == "star":
        return {"kind": "star", "label": "A star", "confidence": "high",
                "why": ["Its size matches the telescope's point-spread function: a point of light, not an extended object."]}
    if det["ellipticity"] >= 0.93 and det["size_px"] > 120 and (det["concentration"] or 0) < 4:
        return {"kind": "artifact", "label": "Possibly an artifact (trail or spike)", "confidence": "low",
                "why": [f"Extremely elongated (ellipticity {det['ellipticity']}) and long ({det['size_px']:.0f} px): typical of a "
                        "satellite trail or a diffraction spike."]}
    if img["emission_line"]:
        gas += 2
        why.append(f"The filter ({img['filter']}) isolates {img['emission_line']}: an image made to show glowing gas.")
    if img["field_kind"] == "nebula":
        gas += 2
        why.append(f"This field is {img['field']}.")
    elif img["field_kind"] in ("deep", "cluster"):
        gas -= 2
        why.append(f"This field is {img['field']}.")
    if det["size_arcsec"] and det["size_arcsec"] > 12:
        gas += 1
        why.append(f"It is big on the sky ({det['size_arcsec']:.0f}″ across); distant galaxies are usually a few arcseconds.")
    c = det["concentration"]
    if c is not None:
        if c < 1.6:
            gas += 2
            why.append(f"Its light is flat, with no bright core (concentration {c:.1f}; galaxies are usually 4 or more).")
        elif c < 2.5:
            gas += 1
            why.append(f"Its light is only weakly concentrated (concentration {c:.1f}).")
        elif c > 6:
            gas -= 2
            why.append(f"Its light is strongly concentrated toward a core (concentration {c:.1f}), as galaxies are.")
    if gas >= 3:
        return {"kind": "nebula", "label": "Probably glowing gas, not a galaxy", "confidence": "high" if gas >= 5 else "medium",
                "why": why + (["Our CNN still calls it a galaxy shape because galaxies are all it knows."] if cnn else [])}
    if cnn and cnn["conf"] is not None and cnn["conf"] < 0.45:
        return {"kind": CLASS_GROUP.get(cnn["label"], "galaxy"), "label": f"Probably a galaxy; shape uncertain", "confidence": "low",
                "why": why + [f"The CNN is unsure: its best guess, {cnn['label']}, has only {cnn['conf'] * 100:.0f}%."]}
    if cnn:
        return {"kind": CLASS_GROUP.get(cnn["label"], "galaxy"), "label": f"Likely a galaxy: {cnn['label'].lower()}",
                "confidence": "high" if cnn["conf"] >= 0.8 and gas <= 0 else "medium",
                "why": why + [f"Our CNN says {CLASS_NOTE.get(cnn['label'], cnn['label'])} ({cnn['conf'] * 100:.0f}%)."]}
    return {"kind": "galaxy", "label": "Likely a galaxy", "confidence": "medium" if gas <= 0 else "low",
            "why": why + ["Extended, with no sign of being gas or a star; the CNN did not classify it (too small or faint)."]}


# ---------- optional: Claude writes the narrative ----------

SYSTEM = """You are the Cosmic Agent inside a live astronomy app. A user clicked an object the app's computer vision \
outlined in a JWST or Hubble image. You get measured facts (JSON) and the app's evidence-based verdict.

Write a short explanation for a curious non-expert: 4 to 6 sentences, plain prose, no headings or bullet points.
- Start with what the object most likely is, consistent with the verdict.
- Ground every specific claim in the facts; use general astronomy knowledge only for background.
- If the verdict says it is not a galaxy, say plainly that the CNN's galaxy label is wrong and why.
- Mention what the filter/wavelength reveals when it matters.
- Never invent numbers (distances, sizes, names) that are not in the facts."""


def _claude_ready():
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
                or os.environ.get("ANTHROPIC_PROFILE") or (Path.home() / ".config" / "anthropic").exists())


def narrative(f, v):
    if not _claude_ready():
        return None
    import anthropic
    try:
        client = anthropic.Anthropic()
        resp = client.beta.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=2000,
            output_config={"effort": "low"},
            system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": json.dumps({"facts": f, "verdict": v}, sort_keys=True)}],
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
    except anthropic.APIStatusError as e:
        log.warning("Claude narrative failed (%s): %s", e.status_code, e.message)
        return None
    except anthropic.APIConnectionError as e:
        log.warning("Claude narrative: connection error %s", e)
        return None
    if resp.stop_reason == "refusal":
        return None
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    return {"text": text, "model": resp.model} if text else None


# ---------- entry point ----------

def explain(con, det_id, use_claude=True):
    cache = CACHE_DIR / f"{det_id}.json"
    f = facts_for(con, det_id)
    if f is None:
        return None
    v = verdict(f)
    kb = KNOWLEDGE.get(v["kind"]) or KNOWLEDGE.get(CLASS_GROUP.get((f["cnn"] or {}).get("label"), ""), None)
    out = {"facts": f, "verdict": v,
           "knowledge": {"title": kb[0], "text": kb[1]} if kb else None,
           "narrative": None, "claude_available": _claude_ready()}
    if use_claude and out["claude_available"]:
        key = json.dumps([f, v], sort_keys=True)
        if cache.exists():
            c = json.loads(cache.read_text(encoding="utf-8"))
            if c.get("key") == key:
                out["narrative"] = c["narrative"]
                return out
        out["narrative"] = narrative(f, v)
        if out["narrative"]:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps({"key": key, "narrative": out["narrative"]}), encoding="utf-8")
    return out
