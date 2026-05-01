"""
vln_dataset.py - PyTorch Dataset wrapping pre-rendered VLN trajectories.

Each trajectory is one .pt file written by render_dataset.py containing:
    rgb         (T, H, W, 3) uint8
    actions     (T,)         long
    instruction str
    + metadata

This class:
    - reads the trajectory files lazily (one per __getitem__ call)
    - exposes train/val splits via a fixed random seed
    - provides a collate_fn that pads variable-length sequences to a
      common length within a batch and produces a `mask` tensor.
"""
from __future__ import annotations
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader


DATA_DIR = Path("data/vln_traj")


@dataclass
class VLNSample:
    rgb:         torch.Tensor   # (T, H, W, 3) uint8
    actions:     torch.Tensor   # (T,) long
    instruction: str
    T:           int
    episode_id:  str
    scene_id:    str


class VLNTrajectoryDataset(Dataset):
    def __init__(self, file_list: List[Path]):
        self.files = file_list

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> VLNSample:
        d = torch.load(self.files[idx], map_location="cpu", weights_only=False)
        return VLNSample(
            rgb        =d["rgb"],
            actions    =d["actions"],
            instruction=d["instruction"],
            T          =int(d["T"]),
            episode_id =str(d["episode_id"]),
            scene_id   =str(d["scene_id"]),
        )


def make_splits(data_dir: Path = DATA_DIR,
                val_frac: float = 0.2,
                seed: int = 42) -> Tuple[List[Path], List[Path]]:
    """Random episode-level train/val split."""
    files = sorted(data_dir.glob("*.pt"))
    rng = random.Random(seed)
    rng.shuffle(files)
    n_val = max(1, int(round(val_frac * len(files))))
    val_files = files[:n_val]
    train_files = files[n_val:]
    return train_files, val_files


def collate_vln(batch: List[VLNSample]) -> dict:
    """Pad variable-length trajectories to the longest T in the batch.

    Returns a dict with:
        rgb        : (B, T_max, H, W, 3) uint8
        actions    : (B, T_max) long
        mask       : (B, T_max) bool   - True where t < real_T
        instructions: list[str], len B
        lengths    : (B,) long
    """
    B = len(batch)
    T_max = max(s.T for s in batch)
    H = W = batch[0].rgb.shape[1]
    C = batch[0].rgb.shape[3]

    rgb     = torch.zeros(B, T_max, H, W, C, dtype=torch.uint8)
    actions = torch.zeros(B, T_max,           dtype=torch.long)
    mask    = torch.zeros(B, T_max,           dtype=torch.bool)
    lengths = torch.zeros(B,                  dtype=torch.long)
    instructions = []

    for i, s in enumerate(batch):
        t = s.T
        rgb[i, :t]     = s.rgb
        actions[i, :t] = s.actions
        mask[i, :t]    = True
        lengths[i]     = t
        instructions.append(s.instruction)

    return {
        "rgb":         rgb,
        "actions":     actions,
        "mask":        mask,
        "lengths":     lengths,
        "instructions": instructions,
    }


def build_loaders(data_dir: Path = DATA_DIR,
                  batch_size: int = 4,
                  val_frac: float = 0.2,
                  num_workers: int = 0,
                  seed: int = 42) -> Tuple[DataLoader, DataLoader, dict]:
    """Build train + val DataLoaders. Returns (train_loader, val_loader, info)."""
    train_files, val_files = make_splits(data_dir, val_frac, seed)
    train_ds = VLNTrajectoryDataset(train_files)
    val_ds   = VLNTrajectoryDataset(val_files)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_vln, num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_vln, num_workers=num_workers,
    )
    info = {
        "n_train":       len(train_ds),
        "n_val":         len(val_ds),
        "train_files":   [str(p.name) for p in train_files],
        "val_files":     [str(p.name) for p in val_files],
    }
    return train_loader, val_loader, info


if __name__ == "__main__":
    print("VLN dataset - smoke test")
    print("=" * 60)

    train_loader, val_loader, info = build_loaders(batch_size=4)
    print(f"  train episodes: {info['n_train']}")
    print(f"  val   episodes: {info['n_val']}")
    print(f"  example train files: {info['train_files'][:3]}")
    print(f"  example val   files: {info['val_files'][:3]}")

    # Get one batch from train loader and inspect shapes
    print("\nFetching one training batch...")
    batch = next(iter(train_loader))
    print(f"  rgb         shape: {tuple(batch['rgb'].shape)}      dtype: {batch['rgb'].dtype}")
    print(f"  actions     shape: {tuple(batch['actions'].shape)}    dtype: {batch['actions'].dtype}")
    print(f"  mask        shape: {tuple(batch['mask'].shape)}    dtype: {batch['mask'].dtype}")
    print(f"  lengths           : {batch['lengths'].tolist()}")
    print(f"  instructions (first):  '{batch['instructions'][0]}'")

    # Verify mask correctness: mask[i, t] is True iff t < lengths[i]
    for i in range(batch['rgb'].shape[0]):
        L = batch['lengths'][i].item()
        true_count = batch['mask'][i].sum().item()
        assert true_count == L, f"mask mismatch at item {i}: {true_count} vs {L}"
    print("\nMask correctness: OK")

    # Action histogram across train set (sanity check on expert distribution)
    print("\nAction distribution in training set (slow, scans all files):")
    counts = torch.zeros(4, dtype=torch.long)
    total_steps = 0
    for f in info['train_files']:
        d = torch.load(DATA_DIR / f, map_location="cpu", weights_only=False)
        for a in d['actions'].tolist():
            counts[a] += 1
        total_steps += int(d['T'])
    pct = counts.float() / max(total_steps, 1) * 100
    names = ["stop", "forward", "left", "right"]
    for n, c, p in zip(names, counts.tolist(), pct.tolist()):
        print(f"  {n:>10s}: {c:5d}  ({p:5.1f}%)")
    print(f"  total steps: {total_steps}")

    print("\nvln_dataset.py smoke test complete.")
