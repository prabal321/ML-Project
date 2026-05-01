"""
eval_per_scene.py - per-scene breakdown of validation accuracy.

Loads the trained model and evaluates per-step accuracy SEPARATELY
on each scene present in the val split. This is a quick proxy for
"performance on unseen environments" — episodes from each scene are
held out from the OTHER scene's perspective.

Usage:
    python code/evaluation/eval_per_scene.py \
      --ckpt code/results/il_runs/concat_lr0.0003_1777577508/best.pt
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from models.vln_model import VLNModel
from data.vln_dataset import VLNTrajectoryDataset, collate_vln
from torch.utils.data import DataLoader

DATA_DIR = Path("data/vln_traj")


def load_model(ckpt_path: Path) -> VLNModel:
    print(f"Loading: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    fusion = ckpt.get("args", {}).get("fusion", "concat")
    m = VLNModel(fusion_kind=fusion, device="cpu")
    m.load_state_dict(ckpt["model_state"])
    m.eval()
    return m


@torch.no_grad()
def evaluate(model, loader, label):
    correct = 0
    total = 0
    for batch in loader:
        rgb = batch["rgb"].float()
        actions = batch["actions"]
        mask = batch["mask"]
        instructions = batch["instructions"]
        logits = model(rgb, instructions)
        preds = logits.argmax(dim=-1)
        c = ((preds == actions) & mask).sum().item()
        t = mask.sum().item()
        correct += c
        total += t
    acc = correct / max(total, 1)
    print(f"  {label:30s}  acc = {acc:.3f}  ({correct}/{total} steps)")
    return acc, correct, total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    args = p.parse_args()

    ckpt_path = Path(args.ckpt)
    model = load_model(ckpt_path)

    # Load val files from the saved split
    split = json.load(open(ckpt_path.parent / "split.json"))
    val_files = [DATA_DIR / fn for fn in split["val_files"]]
    print(f"Total val episodes: {len(val_files)}")

    # Group by scene
    by_scene = defaultdict(list)
    for f in val_files:
        # filename is like "skokloster-castle__ep23.pt"
        scene = f.name.split("__")[0]
        by_scene[scene].append(f)

    print(f"\nVal split per scene:")
    for s, files in by_scene.items():
        print(f"  {s:30s}  n_episodes = {len(files)}")

    print(f"\nPer-scene validation accuracy:")
    results = {}
    overall_correct = 0
    overall_total = 0
    for scene, files in by_scene.items():
        ds = VLNTrajectoryDataset(files)
        loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_vln)
        acc, c, t = evaluate(model, loader, scene)
        results[scene] = {"acc": acc, "n_eps": len(files), "n_steps": t, "correct": c}
        overall_correct += c
        overall_total += t

    overall_acc = overall_correct / max(overall_total, 1)
    print(f"\n  {'(combined val)':30s}  acc = {overall_acc:.3f}  "
          f"({overall_correct}/{overall_total} steps)")

    # Compute the gap (proxy for "scene generalization")
    accs = [r["acc"] for r in results.values()]
    if len(accs) >= 2:
        gap = max(accs) - min(accs)
        print(f"\nPer-scene accuracy gap: {gap:.3f} "
              f"(max {max(accs):.3f} - min {min(accs):.3f})")
        print("Larger gap = more scene-specific overfitting.")

    # Save
    out = ckpt_path.parent / "per_scene_eval.json"
    summary = {
        "ckpt":     str(ckpt_path),
        "overall":  overall_acc,
        "per_scene": results,
        "gap":      max(accs) - min(accs) if len(accs) >= 2 else 0.0,
    }
    open(out, "w").write(json.dumps(summary, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
