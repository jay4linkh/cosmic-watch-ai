"""Train the Mars terrain/scene classifier on NASA's labelled Curiosity images.

    python -m cosmic_watch.ml.mars_train            # RTX 3060: ~10 min

Data: "MSL Curiosity Rover Images with Science and Engineering Classes" v2.1 (Lu & Wagstaff,
NASA/JPL; doi:10.5281/zenodo.4033453): 6,820 Mastcam/MAHLI images, 19 classes (drill hole,
layers, light-toned veins, sand, wheel tracks, close-up rock, distant landscape, rover parts...),
with the official train / val / test split by sol. Unzip it into data/datasets/mars/.

Model: ImageNet-pretrained ConvNeXt-Tiny (timm) fine-tuned at 224 px, class-weighted loss,
mixed precision; validation picks the epoch, the held-out test split is reported once.
Output: models/mars_cnn_v1.onnx + .json model card (classes, metrics, checksum).
"""
import argparse
import hashlib
import json
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ..config import DATA, ROOT

DS = DATA / "datasets" / "mars" / "msl-labeled-data-set-v2.1"
MODEL_DIR = ROOT / "models"
VERSION = "mars_cnn_v1"
SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
ARCH = "convnext_tiny.in12k_ft_in1k"


def classes():
    rows = [l.strip().split(",", 1) for l in (DS / "class_map.csv").read_text().splitlines() if l.strip()]
    return [name for _, name in sorted(rows, key=lambda r: int(r[0]))]


def split(name):
    items = []
    for line in (DS / f"{name}-set-v2.1.txt").read_text().splitlines():
        if line.strip():
            f, k = line.split()
            items.append((f, int(k)))
    return items


def load_rgb(path, size=SIZE):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def to_tensor(rgb_u8):
    x = (rgb_u8.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


class MarsSet(Dataset):
    def __init__(self, items, train):
        self.items, self.train = items, train
        self.cache = {}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        f, k = self.items[i]
        img = load_rgb(DS / "images" / f, 256 if self.train else SIZE)
        if self.train:
            rng = np.random.default_rng()
            y0, x0 = rng.integers(0, 256 - SIZE + 1, 2)
            img = img[y0:y0 + SIZE, x0:x0 + SIZE]
            if rng.random() < 0.5:
                img = img[:, ::-1]
            x = img.astype(np.float32) * rng.uniform(0.75, 1.25) + rng.uniform(-20, 20)   # exposure
            img = np.clip(x, 0, 255).astype(np.uint8)
        return to_tensor(img), k


@torch.no_grad()
def evaluate(model, loader, dev, n_cls):
    model.eval()
    ps, ys = [], []
    for x, y in loader:
        with torch.autocast("cuda", enabled=dev.type == "cuda"):
            ps.append(model(x.to(dev)).float().softmax(1).cpu())
        ys.append(y)
    p, y = torch.cat(ps), torch.cat(ys)
    top = p.argmax(1)
    cm = torch.zeros(n_cls, n_cls, dtype=torch.long)
    for t, q in zip(y, top):
        cm[t, q] += 1
    recall = [cm[i, i].item() / max(cm[i].sum().item(), 1) for i in range(n_cls)]
    return {"accuracy": round((top == y).float().mean().item(), 4),
            "top3_accuracy": round((p.topk(3, 1).indices == y[:, None]).any(1).float().mean().item(), 4),
            "balanced_accuracy": round(float(np.mean([r for i, r in enumerate(recall) if cm[i].sum() > 0])), 4),
            "per_class_recall": recall, "n": int(len(y))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)
    import timm

    names = classes()
    tr, va, te = split("train"), split("val"), split("test")
    print(f"classes {len(names)} · train {len(tr)} · val {len(va)} · test {len(te)}")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(ARCH, pretrained=True, num_classes=len(names), drop_rate=0.3, drop_path_rate=0.1).to(dev)
    counts = np.bincount([k for _, k in tr], minlength=len(names)).astype(np.float32)
    w = torch.tensor((counts.max() / np.maximum(counts, 1)) ** 0.5, device=dev)
    dl_tr = DataLoader(MarsSet(tr, True), batch_size=args.batch, shuffle=True, num_workers=args.workers,
                       pin_memory=True, persistent_workers=args.workers > 0, drop_last=True)
    dl_va = DataLoader(MarsSet(va, False), batch_size=96, num_workers=2)
    dl_te = DataLoader(MarsSet(te, False), batch_size=96, num_workers=2)
    head = [p for n, p in model.named_parameters() if n.startswith("head")]
    body = [p for n, p in model.named_parameters() if not n.startswith("head")]
    opt = torch.optim.AdamW([{"params": body, "lr": args.lr / 4}, {"params": head, "lr": args.lr}], weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr / 4, args.lr], total_steps=args.epochs * len(dl_tr), pct_start=0.15)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    best, best_state, history = -1, None, []
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, run = time.time(), 0.0
        for x, y in dl_tr:
            x, y = x.to(dev, non_blocking=True), y.to(dev)
            with torch.autocast("cuda", enabled=dev.type == "cuda"):
                loss = F.cross_entropy(model(x), y, weight=w, label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += loss.item()
        m = evaluate(model, dl_va, dev, len(names))
        history.append({"epoch": ep, "loss": round(run / len(dl_tr), 4), "val_acc": m["accuracy"], "val_bal": m["balanced_accuracy"]})
        print(f"epoch {ep:2d}/{args.epochs} loss {run / len(dl_tr):.3f} val acc {m['accuracy']:.3f} "
              f"balanced {m['balanced_accuracy']:.3f} top-3 {m['top3_accuracy']:.3f} ({time.time() - t0:.0f}s)", flush=True)
        if m["accuracy"] > best:
            best, best_val = m["accuracy"], m
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    test = evaluate(model, dl_te, dev, len(names))
    print(f"\nheld-out test: acc {test['accuracy']:.3f} · balanced {test['balanced_accuracy']:.3f} · top-3 {test['top3_accuracy']:.3f}")
    model.eval().float().cpu()
    DATA.joinpath("models").mkdir(parents=True, exist_ok=True)
    torch.save(best_state, DATA / "models" / f"{VERSION}.pt")
    onnx_path = MODEL_DIR / f"{VERSION}.onnx"
    torch.onnx.export(model, torch.zeros(1, 3, SIZE, SIZE), onnx_path, input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {0: "n"}, "logits": {0: "n"}}, opset_version=17, dynamo=False)
    sha = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
    fmt_rec = lambda m: {**m, "per_class_recall": {names[i]: round(r, 3) for i, r in enumerate(m["per_class_recall"])}}
    card = {"version": VERSION, "file": onnx_path.name, "sha256": sha, "architecture": f"{ARCH} (ImageNet), fine-tuned",
            "size": SIZE, "mean": MEAN.tolist(), "std": STD.tolist(), "classes": names,
            "data": {"name": "MSL Curiosity Rover Images with Science and Engineering Classes v2.1",
                     "doi": "10.5281/zenodo.4033453", "train": len(tr), "val": len(va), "test": len(te)},
            "val_metrics": fmt_rec(best_val), "test_metrics": fmt_rec(test), "history": history,
            "caveats": "Trained on Curiosity Mastcam/MAHLI images; Perseverance cameras differ (colour, field of view). "
                       "Train/val/test are split by sol, so test accuracy reflects new terrain.",
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}
    (MODEL_DIR / f"{VERSION}.json").write_text(json.dumps(card, indent=2))
    print(f"saved {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
