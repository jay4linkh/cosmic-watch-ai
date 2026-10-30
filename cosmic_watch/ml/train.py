"""Train the galaxy-morphology CNN.

    python -m cosmic_watch.ml.train                 # Galaxy10 DECaLS (+ your Inspect labels if any)
    python -m cosmic_watch.ml.train --epochs 3      # quick run

Data: Galaxy10 DECaLS (17 736 galaxies, 10 classes, labels from Galaxy Zoo volunteers),
https://astronn.readthedocs.io/en/latest/galaxy10.html , downloaded to data/datasets/galaxy10/.

Model: ImageNet-pretrained ResNet-18 fine-tuned on grey 128x128 cut-outs. Our live JWST/Hubble
images are single-filter greyscale JPEG previews, so training converts the colour DECaLS images
to grey and augments with the things that differ between telescopes: any rotation, flips, zoom,
contrast stretch (gamma), PSF blur and noise.

Your own labels from Inspect mode (spiral / elliptical / edge-on / merger) are mixed in with a
"coarse" loss on the summed probabilities of each group, weighted higher than Galaxy10.

Outputs: models/galaxy_cnn_v1.onnx (+ .json model card with metrics and checksum).
"""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

from ..config import DATA, ROOT
from .classes import CLASSES, COARSE
from .cutouts import SIZE, MEAN, STD, cutout

H5 = DATA / "datasets" / "galaxy10" / "Galaxy10_DECals.h5"
CACHE = DATA / "datasets" / "galaxy10" / f"g10_gray_{SIZE}.npz"
MODEL_DIR = ROOT / "models"
VERSION = "galaxy_cnn_v1"
COARSE_NAMES = list(COARSE)


# ---------- data ----------

def load_galaxy10():
    """Galaxy10 as grey SIZE x SIZE uint8 (cached after the first run)."""
    if CACHE.exists():
        z = np.load(CACHE)
        return z["x"], z["y"]
    import h5py
    with h5py.File(H5, "r") as f:
        n = f["ans"].shape[0]
        y = f["ans"][:].astype(np.int64)
        x = np.empty((n, SIZE, SIZE), np.uint8)
        for s in range(0, n, 512):           # stream: the raw file is 3.5 GB in memory
            chunk = f["images"][s:s + 512]
            for i, img in enumerate(chunk):
                g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                x[s + i] = cv2.resize(g, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
            print(f"  prepared {min(s + 512, n)}/{n}", end="\r")
    np.savez_compressed(CACHE, x=x, y=y)
    print()
    return x, y


def load_user_labels():
    """Your Inspect-mode galaxy labels -> (cut-outs, coarse class index)."""
    from .. import db
    con = db.connect()
    rows = con.execute("""SELECT u.key, u.label, d.x, d.y, d.a, i.path, i.stats FROM user_labels u
                          JOIN detections d ON d.id = CAST(substr(u.key, instr(u.key, ':') + 1) AS INTEGER)
                          JOIN images i ON i.id = d.image_id
                          WHERE u.kind = 'galaxy'""").fetchall()
    xs, ys = [], []
    for r in rows:
        if r["label"] not in COARSE_NAMES:
            continue                          # star / artifact: no Galaxy10 class to learn from
        p = Path(r["path"]); p = p if p.is_absolute() else ROOT / p
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        scale = json.loads(r["stats"] or "{}").get("scale", 1.0) or 1.0
        xs.append(cutout(gray, r["x"] / scale, r["y"] / scale, r["a"] / scale))
        ys.append(COARSE_NAMES.index(r["label"]))
    return (np.array(xs, np.uint8), np.array(ys, np.int64)) if xs else (None, None)


def augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Telescope-agnostic augmentation on a uint8 grey image."""
    h, w = img.shape
    ang = rng.uniform(0, 360)
    sc = rng.uniform(0.8, 1.25)
    M = cv2.getRotationMatrix2D((w / 2 + rng.uniform(-4, 4), h / 2 + rng.uniform(-4, 4)), ang, sc)
    out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    if rng.random() < 0.5:
        out = out[:, ::-1]
    x = out.astype(np.float32) / 255.0
    x = np.clip(x, 0, 1) ** rng.uniform(0.55, 1.6)                     # contrast stretch
    x = x * rng.uniform(0.75, 1.3) + rng.uniform(-0.05, 0.05)
    if rng.random() < 0.6:
        s = rng.uniform(0.3, 1.6)                                       # PSF blur
        x = cv2.GaussianBlur(x, (0, 0), s)
    if rng.random() < 0.7:
        x = x + rng.normal(0, rng.uniform(0.005, 0.05), x.shape)        # sky / read noise
    return np.clip(x, 0, 1)


class Cutouts(Dataset):
    def __init__(self, x, y, coarse, train, seed=0):
        self.x, self.y, self.coarse, self.train = x, y, coarse, train
        self.seed = seed

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        rng = np.random.default_rng((self.seed, i, int(time.time() * 1000) % 100000)) if self.train else None
        img = augment(self.x[i], rng) if self.train else self.x[i].astype(np.float32) / 255.0
        t = torch.from_numpy(((img - MEAN) / STD).astype(np.float32))[None].repeat(3, 1, 1)
        return t, int(self.y[i]), int(self.coarse[i])


# ---------- model ----------

def build_model():
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Sequential(nn.Dropout(0.25), nn.Linear(m.fc.in_features, len(CLASSES)))
    return m


GROUP = torch.zeros(len(COARSE), len(CLASSES))
for gi, idx in enumerate(COARSE.values()):
    GROUP[gi, idx] = 1.0


def loss_fn(logits, y, coarse, class_w, smoothing=0.05):
    """Fine-grained CE for Galaxy10 rows (y >= 0); coarse NLL for your labels (y == -1)."""
    total = logits.new_zeros(())
    fine = y >= 0
    if fine.any():
        total = total + F.cross_entropy(logits[fine], y[fine], weight=class_w, label_smoothing=smoothing)
    user = ~fine
    if user.any():
        p = logits[user].softmax(1) @ GROUP.to(logits.device).T
        total = total + 2.0 * F.nll_loss(torch.log(p.clamp_min(1e-6)), coarse[user])
    return total


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    preds, ys = [], []
    for xb, yb, _ in loader:
        with torch.autocast("cuda", enabled=dev.type == "cuda"):
            preds.append(model(xb.to(dev, non_blocking=True)).float().softmax(1).cpu())
        ys.append(yb)
    p, y = torch.cat(preds), torch.cat(ys)
    top = p.argmax(1)
    cm = torch.zeros(len(CLASSES), len(CLASSES), dtype=torch.long)
    for t, q in zip(y, top):
        cm[t, q] += 1
    coarse_p = (p @ GROUP.T).argmax(1)
    coarse_y = (GROUP.T[y]).argmax(1)
    top2 = (p.topk(2, 1).indices == y[:, None]).any(1).float().mean().item()
    return {
        "accuracy": round((top == y).float().mean().item(), 4),
        "top2_accuracy": round(top2, 4),
        "coarse_accuracy": round((coarse_p == coarse_y).float().mean().item(), 4),
        "per_class_recall": {CLASSES[i]: round(cm[i, i].item() / max(cm[i].sum().item(), 1), 3) for i in range(len(CLASSES))},
        "confusion": cm.tolist(),
    }


# ---------- main ----------

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args(argv)

    torch.manual_seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev} {torch.cuda.get_device_name(0) if dev.type == 'cuda' else ''}")

    x, y = load_galaxy10()
    print(f"Galaxy10: {len(x)} images, classes {np.bincount(y).tolist()}")
    rng = np.random.default_rng(0)
    val_idx = np.concatenate([rng.permutation(np.flatnonzero(y == c))[: max(1, int(0.15 * (y == c).sum()))] for c in range(len(CLASSES))])
    tr_mask = np.ones(len(y), bool); tr_mask[val_idx] = False
    coarse_y = np.array([[i for i, g in enumerate(COARSE.values()) if c in g][0] for c in y])

    xtr, ytr, ctr = x[tr_mask], y[tr_mask], coarse_y[tr_mask]
    ux, uy = load_user_labels()
    n_user = 0 if ux is None else len(ux)
    if n_user:
        # repeat your labels so they matter next to 15k Galaxy10 images
        rep = max(1, 300 // n_user)
        xtr = np.concatenate([xtr, np.repeat(ux, rep, 0)])
        ytr = np.concatenate([ytr, np.full(n_user * rep, -1)])
        ctr = np.concatenate([ctr, np.repeat(uy, rep, 0)])
    print(f"train {len(xtr)} (your labels: {n_user}) · val {len(val_idx)}")

    tr = DataLoader(Cutouts(xtr, ytr, ctr, True), batch_size=args.batch, shuffle=True, num_workers=args.workers,
                    pin_memory=True, persistent_workers=args.workers > 0, drop_last=True)
    va = DataLoader(Cutouts(x[val_idx], y[val_idx], coarse_y[val_idx], False), batch_size=256, num_workers=2, pin_memory=True)

    counts = np.bincount(y[tr_mask], minlength=len(CLASSES))
    class_w = torch.tensor((counts.max() / counts) ** 0.5, dtype=torch.float32, device=dev)   # gentle re-balancing
    model = build_model().to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(tr), pct_start=0.15)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")

    best, best_state, history = -1, None, []
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, run, nb = time.time(), 0.0, 0
        for xb, yb, cb in tr:
            xb = xb.to(dev, non_blocking=True).to(memory_format=torch.channels_last)
            yb, cb = yb.to(dev), cb.to(dev)
            with torch.autocast("cuda", enabled=dev.type == "cuda"):
                loss = loss_fn(model(xb), yb, cb, class_w)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += loss.item(); nb += 1
        m = evaluate(model, va, dev)
        history.append({"epoch": ep, "loss": round(run / nb, 4), **{k: m[k] for k in ("accuracy", "top2_accuracy", "coarse_accuracy")}})
        print(f"epoch {ep:2d}/{args.epochs}  loss {run / nb:.3f}  val acc {m['accuracy']:.3f}  top-2 {m['top2_accuracy']:.3f}  "
              f"coarse {m['coarse_accuracy']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
        if m["accuracy"] > best:
            best, best_metrics = m["accuracy"], m
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval().float().to("cpu").to(memory_format=torch.contiguous_format)
    DATA.joinpath("models").mkdir(parents=True, exist_ok=True)
    torch.save(best_state, DATA / "models" / f"{VERSION}.pt")

    # ONNX for light-weight inference in the pipeline (no PyTorch needed at runtime)
    MODEL_DIR.mkdir(exist_ok=True)
    onnx_path = MODEL_DIR / f"{VERSION}.onnx"
    torch.onnx.export(model, torch.zeros(1, 3, SIZE, SIZE), onnx_path, input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {0: "n"}, "logits": {0: "n"}}, opset_version=17, dynamo=False)
    sha = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
    card = {
        "version": VERSION, "sha256": sha, "file": onnx_path.name,
        "architecture": "ResNet-18 (ImageNet-pretrained), fine-tuned", "input": f"grey {SIZE}x{SIZE}, replicated to 3 channels",
        "normalize": {"mean": MEAN, "std": STD}, "classes": CLASSES, "coarse": COARSE,
        "data": {"galaxy10_decals": int(len(x)), "train": int(tr_mask.sum()), "val": int(len(val_idx)), "your_labels": n_user},
        "training": {"epochs": args.epochs, "batch": args.batch, "lr": args.lr, "device": str(dev),
                     "augmentation": "rotation, flip, zoom, gamma stretch, PSF blur, noise (telescope-agnostic, greyscale)"},
        "val_metrics": best_metrics, "history": history,
        "caveats": ("Trained on DECaLS ground-based colour images converted to grey; our live images are JWST/HST "
                    "single-filter previews, so expect lower accuracy there, especially for small or faint sources."),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }
    (MODEL_DIR / f"{VERSION}.json").write_text(json.dumps(card, indent=2))
    print(f"\nbest val accuracy {best:.3f} · saved {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB) sha256 {sha[:12]}…")


if __name__ == "__main__":
    main()
