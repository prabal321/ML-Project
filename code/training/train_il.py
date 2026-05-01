"""
train_il.py - imitation learning (behavior cloning) for the VLN model.

Loads pre-rendered expert trajectories from data/vln_traj/, trains the
VLN model end-to-end (frozen CLIP, trainable fusion+GRU+policy) to
mimic the expert's actions.

Run from the habitat-lab folder:
    python code/training/train_il.py
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# Make our local packages importable
ROOT = Path(__file__).resolve().parents[2]   # .../habitat-lab
sys.path.insert(0, str(ROOT / "code"))
from models.vln_model import VLNModel, NUM_ACTIONS
from data.vln_dataset import build_loaders


def evaluate(model: VLNModel, loader, device: torch.device,
             max_batches: int = None) -> dict:
    """Run the model on `loader` and return loss + per-step accuracy."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total   = 0
    n_batches = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            rgb         = batch["rgb"].to(device).float()       # (B, T, H, W, 3)
            actions     = batch["actions"].to(device)           # (B, T)
            mask        = batch["mask"].to(device)              # (B, T)
            instructions = batch["instructions"]

            logits = model(rgb, instructions)                   # (B, T, A)
            B, T, A = logits.shape
            loss = F.cross_entropy(
                logits.reshape(B * T, A),
                actions.reshape(B * T),
                reduction="none",
            ).reshape(B, T)
            loss = (loss * mask.float()).sum() / mask.sum().clamp(min=1)

            preds = logits.argmax(dim=-1)                       # (B, T)
            correct_mask = (preds == actions) & mask
            correct += correct_mask.sum().item()
            total   += mask.sum().item()

            total_loss += loss.item()
            n_batches += 1

    model.train()
    return {
        "loss":     total_loss / max(n_batches, 1),
        "accuracy": correct / max(total, 1),
        "n_steps":  total,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--fusion", type=str, default="concat",
                   choices=["concat", "gated", "crossattn"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-frac", type=float, default=1.0,
                   help="Fraction of training data to use (e.g. 0.5 for half).")
    p.add_argument("--scene-filter", type=str, default=None,
                   help="Only use trajectories whose filename contains this string (e.g. 'skokloster-castle' or 'van-gogh-room'). Train uses matched, val uses unmatched.")
    p.add_argument("--finetune-clip", type=int, default=0,
                   help="If > 0, unfreeze the last N transformer layers of CLIP "
                        "(both vision and text) plus projection heads. "
                        "Default 0 means CLIP is fully frozen.")
    p.add_argument("--use-class-weights", action="store_true",
                   help="Reweight loss by inverse class frequency.")
    p.add_argument("--out-dir", type=str,
                   default="code/results/il_runs")
    p.add_argument("--run-name", type=str, default=None,
                   help="Subfolder name; default = fusion+lr+timestamp.")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cpu")

    # --- output dir ---
    run_name = args.run_name or f"{args.fusion}_lr{args.lr:g}_{int(time.time())}"
    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {out_dir}")

    with open(out_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # --- data ---
    print("Building data loaders...")
    train_loader, val_loader, info = build_loaders(
        batch_size=args.batch_size, val_frac=0.2, seed=args.seed,
    )

    # Reduced-data study: subset training files
    if args.data_frac < 1.0:
        from torch.utils.data import DataLoader
        from data.vln_dataset import VLNTrajectoryDataset, collate_vln
        import random as _random
        full = list(train_loader.dataset.files)
        rng = _random.Random(args.seed)
        rng.shuffle(full)
        n_keep = max(1, int(round(args.data_frac * len(full))))
        kept = full[:n_keep]
        train_loader = DataLoader(
            VLNTrajectoryDataset(kept), batch_size=args.batch_size,
            shuffle=True, collate_fn=collate_vln,
        )
        info["n_train_used"] = n_keep
        info["data_frac"]    = args.data_frac
        print(f"  REDUCED DATA: using {n_keep} / {len(full)} training trajectories "
              f"({args.data_frac:.0%})")
    print(f"  n_train={info['n_train']}  n_val={info['n_val']}")

    # Sub-item 1: scene-filtered split (train one scene, val on the other)
    if args.scene_filter is not None:
        from torch.utils.data import DataLoader as _DL
        from data.vln_dataset import VLNTrajectoryDataset as _DS
        from data.vln_dataset import collate_vln as _coll
        wanted = args.scene_filter
        all_files = list(train_loader.dataset.files) + list(val_loader.dataset.files)
        train_files_f = [f for f in all_files if wanted in str(f)]
        val_files_f   = [f for f in all_files if wanted not in str(f)]
        train_loader = _DL(_DS(train_files_f), batch_size=args.batch_size,
                           shuffle=True, collate_fn=_coll)
        val_loader   = _DL(_DS(val_files_f),   batch_size=args.batch_size,
                           shuffle=False, collate_fn=_coll)
        info["scene_filter"] = wanted
        info["n_train_used"] = len(train_files_f)
        info["n_val_used"]   = len(val_files_f)
        print(f"  SCENE FILTER: train on '{wanted}' only ({len(train_files_f)} episodes);"
              f" eval on UNSEEN scenes ({len(val_files_f)} episodes)")
    with open(out_dir / "split.json", "w") as f:
        json.dump(info, f, indent=2)

    # --- class weights (optional) ---
    class_weights = None
    if args.use_class_weights:
        counts = torch.zeros(NUM_ACTIONS, dtype=torch.float)
        for f_path in info["train_files"]:
            d = torch.load(Path("data/vln_traj") / f_path,
                          map_location="cpu", weights_only=False)
            for a in d["actions"].tolist():
                counts[a] += 1
        # inverse frequency, normalized to sum to NUM_ACTIONS
        weights = (counts.sum() / (counts + 1e-6))
        weights = weights / weights.sum() * NUM_ACTIONS
        class_weights = weights.to(device)
        print(f"  class weights: {class_weights.tolist()}")

    # --- model ---
    print(f"\nBuilding VLNModel (fusion={args.fusion})...")
    model = VLNModel(fusion_kind=args.fusion, device="cpu")

    # Optional partial fine-tuning of CLIP
    if args.finetune_clip > 0:
        N = args.finetune_clip
        clip_model = model.encoders.clip
        unfrozen = 0
        # Unfreeze last N layers of vision encoder
        v_layers = clip_model.vision_model.encoder.layers
        for layer in v_layers[-N:]:
            for p in layer.parameters():
                p.requires_grad_(True); unfrozen += p.numel()
        # Unfreeze last N layers of text encoder
        t_layers = clip_model.text_model.encoder.layers
        for layer in t_layers[-N:]:
            for p in layer.parameters():
                p.requires_grad_(True); unfrozen += p.numel()
        # Unfreeze projection heads
        for p in clip_model.visual_projection.parameters():
            p.requires_grad_(True); unfrozen += p.numel()
        for p in clip_model.text_projection.parameters():
            p.requires_grad_(True); unfrozen += p.numel()
        # Set CLIP back to train() so the unfrozen layers actually train
        clip_model.train()
        print(f"  FINE-TUNING CLIP: unfroze last {N} layers + projection heads "
              f"({unfrozen:,} extra trainable params)")
    print(f"  trainable params: {model.trainable_param_count():,}")

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
    )

    writer = SummaryWriter(log_dir=str(out_dir / "tb"))

    # --- training loop ---
    print(f"\nStarting training: {args.epochs} epochs\n" + "-" * 60)
    history = []
    best_val_acc = 0.0
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_steps = 0

        for batch in train_loader:
            rgb         = batch["rgb"].float()                  # (B, T, H, W, 3)
            actions     = batch["actions"]                      # (B, T)
            mask        = batch["mask"]                         # (B, T)
            instructions = batch["instructions"]

            logits = model(rgb, instructions)                   # (B, T, A)
            B, T, A = logits.shape
            loss_per = F.cross_entropy(
                logits.reshape(B * T, A),
                actions.reshape(B * T),
                weight=class_weights,
                reduction="none",
            ).reshape(B, T)
            loss = (loss_per * mask.float()).sum() / mask.sum().clamp(min=1)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()

            preds = logits.argmax(dim=-1)
            correct_mask = (preds == actions) & mask
            epoch_correct += correct_mask.sum().item()
            epoch_steps   += mask.sum().item()
            epoch_loss    += loss.item()
            global_step   += 1

            writer.add_scalar("train/loss_step", loss.item(), global_step)

        train_loss = epoch_loss / max(len(train_loader), 1)
        train_acc  = epoch_correct / max(epoch_steps, 1)

        # --- validation ---
        val_metrics = evaluate(model, val_loader, device)
        elapsed = time.time() - t0

        writer.add_scalar("train/loss_epoch",    train_loss, epoch)
        writer.add_scalar("train/accuracy",      train_acc,  epoch)
        writer.add_scalar("val/loss",            val_metrics["loss"],     epoch)
        writer.add_scalar("val/accuracy",        val_metrics["accuracy"], epoch)

        line = (f"epoch {epoch:3d}/{args.epochs}  "
                f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
                f"val_loss={val_metrics['loss']:.4f}  val_acc={val_metrics['accuracy']:.3f}  "
                f"({elapsed:.1f}s)")
        print(line)
        history.append({
            "epoch":      epoch,
            "train_loss": train_loss,
            "train_acc":  train_acc,
            "val_loss":   val_metrics["loss"],
            "val_acc":    val_metrics["accuracy"],
            "elapsed_s":  elapsed,
        })

        # Save checkpoints
        ckpt = {
            "model_state":  model.state_dict(),
            "args":         vars(args),
            "epoch":        epoch,
            "val_acc":      val_metrics["accuracy"],
        }
        torch.save(ckpt, out_dir / "latest.pt")
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            torch.save(ckpt, out_dir / "best.pt")

        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    writer.close()
    print("-" * 60)
    print(f"Done. Best val accuracy: {best_val_acc:.3f}")
    print(f"Checkpoints + logs in: {out_dir}/")


if __name__ == "__main__":
    main()
