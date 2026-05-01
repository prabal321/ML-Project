"""
vln_model.py - the full Vision-Language Navigation model.

Architecture:
    Frozen CLIP visual encoder (image -> 512-d)
    Frozen CLIP text encoder   (instruction -> 512-d)
    Trainable fusion module    ((512-d, 512-d) -> 512-d)
    Trainable 1-layer GRU      (carries memory across timesteps)
    Trainable policy head      (Linear 512 -> num_actions)

Action space (matches Habitat PointNav):
    0: stop
    1: move_forward (0.25 m)
    2: turn_left   (10 deg)
    3: turn_right  (10 deg)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from encoders import CLIPEncoders, FEATURE_DIM
from fusion import build_fusion


NUM_ACTIONS = 4
ACTION_NAMES = ["stop", "move_forward", "turn_left", "turn_right"]


@dataclass
class VLNStep:
    """One forward-pass output."""
    action_logits: torch.Tensor   # (B, NUM_ACTIONS)
    hidden:        torch.Tensor   # (1, B, hidden_dim) - GRU state to pass to next step


class VLNModel(nn.Module):
    def __init__(self,
                 fusion_kind: str = "concat",
                 hidden_dim: int = 512,
                 num_actions: int = NUM_ACTIONS,
                 device: str = "cpu"):
        super().__init__()
        self.encoders = CLIPEncoders(device=device)
        self.fusion   = build_fusion(fusion_kind, dim=FEATURE_DIM)
        self.gru      = nn.GRU(input_size=FEATURE_DIM,
                               hidden_size=hidden_dim,
                               num_layers=1,
                               batch_first=True)
        self.policy   = nn.Linear(hidden_dim, num_actions)

        self.hidden_dim  = hidden_dim
        self.num_actions = num_actions
        self.device      = torch.device(device)
        self.to(self.device)

    # -------- helpers --------
    def init_hidden(self, batch_size: int) -> torch.Tensor:
        """Initial GRU hidden state (zeros)."""
        return torch.zeros(1, batch_size, self.hidden_dim, device=self.device)

    def encode_text_once(self, instructions: List[str]) -> torch.Tensor:
        """Encode instructions once per episode; cache and reuse."""
        return self.encoders.encode_text(instructions)   # (B, 512)

    # -------- forward (single timestep) --------
    def step(self,
             rgb: torch.Tensor,                # (B, H, W, 3) or (B, 3, H, W)
             text_feat: torch.Tensor,          # (B, 512), pre-cached
             hidden:    torch.Tensor,          # (1, B, hidden_dim)
             ) -> VLNStep:
        """
        Run one timestep: encode current frame, fuse with cached text,
        update GRU, predict action.
        """
        v = self.encoders.encode_image(rgb)            # (B, 512)
        fused = self.fusion(v, text_feat)              # (B, 512)
        gru_in = fused.unsqueeze(1)                    # (B, 1, 512)
        gru_out, new_hidden = self.gru(gru_in, hidden) # (B, 1, H), (1, B, H)
        logits = self.policy(gru_out.squeeze(1))       # (B, num_actions)
        return VLNStep(action_logits=logits, hidden=new_hidden)

    # -------- forward (whole trajectory) --------
    def forward(self,
                rgb_seq:        torch.Tensor,      # (B, T, H, W, 3) or (B, T, 3, H, W)
                instructions:   List[str],         # len B
                ) -> torch.Tensor:
        """
        Run an entire trajectory of length T. Used for training (teacher forcing).
        Returns logits of shape (B, T, num_actions).
        """
        B, T = rgb_seq.shape[0], rgb_seq.shape[1]
        text_feat = self.encode_text_once(instructions)            # (B, 512)
        hidden = self.init_hidden(B)
        all_logits = []
        for t in range(T):
            step_out = self.step(rgb_seq[:, t], text_feat, hidden)
            all_logits.append(step_out.action_logits)
            hidden = step_out.hidden
        return torch.stack(all_logits, dim=1)                      # (B, T, A)

    # -------- parameter counting --------
    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def frozen_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)


if __name__ == "__main__":
    # Smoke test
    import time

    print("Building VLNModel (concat fusion)...")
    t0 = time.time()
    model = VLNModel(fusion_kind="concat", device="cpu")
    print(f"  built in {time.time()-t0:.1f}s")
    print(f"  trainable params: {model.trainable_param_count():,}")
    print(f"  frozen    params: {model.frozen_param_count():,}")

    # Single-step forward
    print("\n[Test 1] single step forward")
    B = 2
    rgb = torch.randint(0, 255, (B, 224, 224, 3), dtype=torch.uint8).float()
    instr = ["walk forward 5 meters", "turn left and stop"]
    text_feat = model.encode_text_once(instr)
    hidden = model.init_hidden(B)
    out = model.step(rgb, text_feat, hidden)
    print(f"  action_logits.shape = {tuple(out.action_logits.shape)}  (expected ({B}, {NUM_ACTIONS}))")
    print(f"  hidden.shape        = {tuple(out.hidden.shape)}        (expected (1, {B}, 512))")

    # Trajectory forward
    print("\n[Test 2] full trajectory forward (T=5)")
    T = 5
    rgb_seq = torch.randint(0, 255, (B, T, 224, 224, 3), dtype=torch.uint8).float()
    t0 = time.time()
    logits = model(rgb_seq, instr)
    elapsed = time.time() - t0
    print(f"  logits.shape = {tuple(logits.shape)}  (expected ({B}, {T}, {NUM_ACTIONS}))")
    print(f"  forward time = {elapsed:.2f}s for {B*T} frames "
          f"({B*T/elapsed:.1f} frames/sec)")

    # Trainable submodule comparison
    print("\n[Test 3] trainable params per submodule")
    for name in ["fusion", "gru", "policy"]:
        sub = getattr(model, name)
        n = sum(p.numel() for p in sub.parameters() if p.requires_grad)
        print(f"  {name:>8s}: {n:,}")

    # Quick fusion-variant comparison
    print("\n[Test 4] swap fusion to crossattn -> param count diff")
    model2 = VLNModel(fusion_kind="crossattn", device="cpu")
    print(f"  concat    trainable: {model.trainable_param_count():,}")
    print(f"  crossattn trainable: {model2.trainable_param_count():,}")

    print("\nvln_model.py smoke test complete.")
