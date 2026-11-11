"""Rebuild the figures in paper/figures from the project's own outputs.

    python paper/make_figures.py

Needs the local database (data/cosmic_watch.db), the model cards in models/ and the
science catalogue in data/science/catalogs/. Nothing here is simulated: every number
is read from a file the pipeline wrote.
"""
import json
import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, Rectangle
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.05})

COARSE = {
    "Round smooth": "elliptical", "In-between round smooth": "elliptical", "Cigar-shaped smooth": "elliptical",
    "Barred spiral": "spiral", "Unbarred tight spiral": "spiral", "Unbarred loose spiral": "spiral",
    "Edge-on without bulge": "edge-on", "Edge-on with bulge": "edge-on",
    "Merging": "merger / disturbed", "Disturbed": "merger / disturbed", "Not a galaxy": "not a galaxy",
}
COLOURS = {"elliptical": "#ffb347", "spiral": "#4fc3f7", "edge-on": "#b388ff",
           "merger / disturbed": "#ff5c8a", "not a galaxy": "#9e9e9e"}


def fig_screens():
    """Figure 1: four views of the running application (frames of the screen recording)."""
    names = [("observatory.jpg", "a  Observatory: every feed and detector"),
             ("inspect.jpg", "b  Inspect: CNN probabilities and a 3D model"),
             ("sun.jpg", "c  The Sun (SDO/AIA) with detected active regions"),
             ("mars.jpg", "d  Perseverance image with rock candidates")]
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 4.3))
    for ax, (f, title) in zip(axs.flat, names):
        im = Image.open(OUT / "screens" / f).convert("RGB")
        w, h = im.size
        ax.imshow(im.crop((0, int(h * 0.04), w, int(h * 0.96))))
        ax.set_title(title, loc="left", fontsize=8.5)
        ax.axis("off")
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "fig1_app.png")
    plt.close(fig)


def fig_deepfield(image_id=243):
    """Figure 2: a public JWST image with every CNN-classified detection outlined by coarse class."""
    db = sqlite3.connect(f"file:{ROOT / 'data' / 'cosmic_watch.db'}?mode=ro", uri=True)
    path, stats, title = db.execute("select path, stats, title from images where id=?", (image_id,)).fetchone()
    scale = json.loads(stats)["scale"]                  # detector coordinates -> preview pixels
    img = np.asarray(Image.open(ROOT / path).convert("L"), dtype=np.float32)
    rows = db.execute("select x, y, a, b, theta, cnn_label from detections where image_id=? and cnn_label is not null",
                      (image_id,)).fetchall()
    det = [(x / scale, y / scale, a / scale, b / scale, th, lab) for x, y, a, b, th, lab in rows]

    xs, ys = np.array([d[0] for d in det]), np.array([d[1] for d in det])
    half = 650                                          # inset: the 1300 px box holding the most detections
    best = max(((cx, cy) for cx, cy in zip(xs, ys)),
               key=lambda c: ((abs(xs - c[0]) < half) & (abs(ys - c[1]) < half)).sum())
    x0 = int(np.clip(best[0] - half, 0, img.shape[1] - 2 * half))
    y0 = int(np.clip(best[1] - half, 0, img.shape[0] - 2 * half))

    lo, hi = np.percentile(img, [1, 99.7])
    disp = np.clip((img - lo) / (hi - lo), 0, 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.75), gridspec_kw={"width_ratios": [1, 1]})
    a1.imshow(disp[::4, ::4], cmap="gray", origin="upper")
    a1.add_patch(Rectangle((x0 / 4, y0 / 4), 2 * half / 4, 2 * half / 4, fill=False, ec="#ffd166", lw=1.2))
    a1.set_title(f"{title.split(': ')[-1]} (JWST NIRCam F150W), full field", fontsize=8.5, loc="left")
    a2.imshow(disp[y0:y0 + 2 * half, x0:x0 + 2 * half], cmap="gray", origin="upper")
    seen = set()
    for x, y, a, b, th, lab in det:
        if not (x0 <= x < x0 + 2 * half and y0 <= y < y0 + 2 * half):
            continue
        c = COARSE.get(lab, "not a galaxy")
        seen.add(c)
        a2.add_patch(Ellipse((x - x0, y - y0), max(6 * a, 18), max(6 * b, 18), angle=np.degrees(th),
                             fill=False, ec=COLOURS[c], lw=1.0))
    a2.set_title("inset: outlines coloured by the galaxy CNN's call", fontsize=8.5, loc="left")
    for ax in (a1, a2):
        ax.set_xticks([]); ax.set_yticks([])
    handles = [plt.Line2D([], [], color=COLOURS[c], lw=2, label=c) for c in COLOURS if c in seen]
    a2.legend(handles=handles, loc="lower right", fontsize=7, framealpha=0.75)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "fig2_deepfield.png")
    plt.close(fig)
    return len(rows)


def fig_confusion():
    """Figure 3: galaxy CNN v2 confusion matrix on the 2,656-galaxy Galaxy10 DECaLS validation split."""
    card = json.loads((ROOT / "models" / "galaxy_cnn_v2.json").read_text())
    cm = np.array(card["val_metrics"]["confusion"], dtype=float)
    norm = cm / cm.sum(1, keepdims=True)
    labels = card["classes"]
    fig, ax = plt.subplots(figsize=(5.6, 4.7))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if norm[i, j] >= 0.02:
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(labels)), labels, rotation=55, ha="right", fontsize=7)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("Galaxy Zoo label")
    m = card["val_metrics"]
    ax.set_title(f"accuracy {m['accuracy']:.1%} · top-2 {m['top2_accuracy']:.1%} · coarse {m['coarse_accuracy']:.1%}",
                 fontsize=8.5)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="fraction of row")
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "fig3_confusion.png")
    plt.close(fig)


def fig_completeness(cat="7_sci-det-1.json"):
    """Figure 4: injection-recovery completeness of the FITS detector, per size class."""
    c = json.loads((ROOT / "data" / "science" / "catalogs" / cat).read_text())
    comp = c["completeness"]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for name, k in comp["classes"].items():
        ax.plot(k["mags"], k["recovered"], "o-", ms=3.5, lw=1.4,
                label=f"{name} (r$_{{50}}$ = {k['r50_px']:.1f} px, 50% at {k['mag_50']:.1f})")
    ax.axhline(0.5, color="0.6", lw=0.8, ls="--")
    ax.axvline(comp["mag_5sigma_point"], color="0.3", lw=0.8, ls=":")
    ax.text(comp["mag_5sigma_point"] - 0.08, 0.97, "5σ point-source\ndepth", fontsize=7, va="top", ha="right")
    m = c["meta"]
    ax.set_title(f"{m['telescope']} {m['instrument']} {m['filter']}, {c['n_sources']:,} sources", fontsize=8.5)
    ax.set_xlabel("injected AB magnitude")
    ax.set_ylabel("fraction recovered")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(fontsize=6.5, frameon=False, loc="lower left")
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "fig4_completeness.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_screens()
    n = fig_deepfield()
    fig_confusion()
    fig_completeness()
    print("figures written to", OUT, "| CNN-classified detections in the JWST field:", n)
