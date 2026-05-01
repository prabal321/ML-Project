"""
eval_paraphrase.py - per-step accuracy with a different synthetic instruction
                     per episode (paraphrasing test for Task 4 sub-item 2).

For each val episode, we generate a NEW synthetic instruction (different RNG
seed -> different template + direction phrasing), then re-evaluate per-step
action accuracy under teacher forcing (i.e., model sees expert frames).

Compares:
    Baseline:   original instruction (from training)  -> existing val_acc
    Paraphrased: new instruction with a fresh seed

Usage:
    python code/evaluation/eval_paraphrase.py \
      --ckpt code/results/il_runs/concat_lr0.0003_1777577508/best.pt
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from models.vln_model import VLNModel, NUM_ACTIONS
from data.instructions import (
    Episode as InstrEpisode, generate_instruction
)
from data.vln_dataset import VLNTrajectoryDataset, make_splits, collate_vln
from torch.utils.data import DataLoader

DATA_DIR = Path("data/vln_traj")


def load_model(ckpt_path: Path) -> VLNModel:
    print(f"Loading: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    fusion = ckpt.get("args", {}).get("fusion", "concat")
    m = VLNModel(fusion_kind=fusion, device="cpu")
    m.load_state_dict(ckpt["model_state"])
    m.eval()
    print(f"  fusion={fusion}, val_acc at save={ckpt.get('val_acc', '?')}")
    return m


def make_paraphrased_loader(val_files, batch_size=4, seed_offset=999):
    """A val DataLoader with one paraphrased instruction per episode."""
    class ParaphrasedDataset(VLNTrajectoryDataset):
        def __getitem__(self, idx):
            sample = super().__getitem__(idx)
            # Use the trajectory file path as a deterministic seed source
            f = self.files[idx]
            d = torch.load(f, map_location="cpu", weights_only=False)
            scene = d["scene_id"]
            sp = tuple(d["start_pos"])
            gp = tuple(d["goal_pos"])
            eid = d["episode_id"]
            # Deterministic but DIFFERENT seed than what was used originally
            import random
            rng = random.Random(hash(eid) & 0xFFFF + seed_offset)
            new_instr = generate_instruction(InstrEpisode(
                scene_id=scene, start_position=sp, goal_position=gp,
                episode_id=str(eid),
            ), rng=rng)
            sample.instruction = new_instr
            return sample

    ds = ParaphrasedDataset(val_files)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      collate_fn=collate_vln)


@torch.no_grad()
def evaluate(model, loader, label):
    correct = 0
    total = 0
    examples = []
    for batch in loader:
        rgb = batch["rgb"].float()
        actions = batch["actions"]
        mask = batch["mask"]
        instructions = batch["instructions"]
        if not examples:
            examples = instructions[:3]   # keep first 3 instructions for display
        logits = model(rgb, instructions)
        preds = logits.argmax(dim=-1)
        c = ((preds == actions) & mask).sum().item()
        t = mask.sum().item()
        correct += c
        total += t
    acc = correct / max(total, 1)
    print(f"\n[{label}]")
    print(f"  per-step val accuracy: {acc:.3f}  ({correct}/{total} steps)")
    print(f"  example instructions:")
    for i, s in enumerate(examples):
        print(f"    {i+1}. {s}")
    return acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    ckpt_path = Path(args.ckpt)
    model = load_model(ckpt_path)

    # Reload the same train/val split as training
    split_path = ckpt_path.parent / "split.json"
    if split_path.exists():
        split = json.load(open(split_path))
        val_files = [DATA_DIR / fn for fn in split["val_files"]]
    else:
        # Fallback: reproduce split with same seed
        _, val_files = make_splits(seed=args.seed)
    print(f"Val episodes: {len(val_files)}")

    # 1. Original instructions
    base_loader = DataLoader(
        VLNTrajectoryDataset(val_files), batch_size=4, shuffle=False,
        collate_fn=collate_vln,
    )
    acc_orig = evaluate(model, base_loader, "Original instructions")

    # 2. Paraphrased instructions
    para_loader = make_paraphrased_loader(val_files, batch_size=4, seed_offset=999)
    acc_para = evaluate(model, para_loader, "Paraphrased instructions")

    # 3. Compare
    print("\n" + "=" * 60)
    print("Paraphrasing test summary")
    print(f"  Original     val acc: {acc_orig:.3f}")
    print(f"  Paraphrased  val acc: {acc_para:.3f}")
    print(f"  Delta:                {acc_para - acc_orig:+.3f}")
    if abs(acc_para - acc_orig) < 0.01:
        verdict = "Robust to paraphrasing (delta < 0.01)"
    elif acc_para < acc_orig - 0.02:
        verdict = "Some sensitivity to phrasing"
    else:
        verdict = "Mostly robust"
    print(f"  Verdict:              {verdict}")

    # Save
    out = ckpt_path.parent / "paraphrase_eval.json"
    summary = {
        "ckpt":         str(ckpt_path),
        "n_val":        len(val_files),
        "orig_val_acc": float(acc_orig),
        "para_val_acc": float(acc_para),
        "delta":        float(acc_para - acc_orig),
        "verdict":      verdict,
    }
    open(out, "w").write(json.dumps(summary, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
