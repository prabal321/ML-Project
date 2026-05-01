"""
fusion.py - cross-modal fusion modules for vision-language navigation.

Provides three fusion strategies, all with the same interface:
    forward(visual_feat: (B, D), text_feat: (B, D))  ->  (B, D_fused)

Variants:
    1. ConcatMLPFusion   - concat + MLP  (baseline)
    2. GatedFusion       - element-wise gated combination
    3. CrossAttnFusion   - multi-head cross-attention (visual queries text)

Default: ConcatMLPFusion. Switch by name in build_fusion().
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConcatMLPFusion(nn.Module):
    """concat([v; t]) -> Linear -> ReLU -> Linear   (the baseline)."""

    def __init__(self, dim: int = 512, hidden_dim: int | None = None,
                 dropout: float = 0.1):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.net = nn.Sequential(
            nn.Linear(2 * dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        self.out_dim = dim

    def forward(self, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = torch.cat([v, t], dim=-1)         # (B, 2*dim)
        return self.net(x)                    # (B, dim)


class GatedFusion(nn.Module):
    """Per-dim gate g in [0,1]; output = g*v + (1-g)*t. Tiny, expressive."""

    def __init__(self, dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.Sigmoid(),
        )
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.out_dim = dim

    def forward(self, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        g = self.gate(torch.cat([v, t], dim=-1))      # (B, dim)
        mixed = g * v + (1 - g) * t                   # (B, dim)
        return self.proj(mixed)                       # (B, dim)


class CrossAttnFusion(nn.Module):
    """Multi-head cross-attention: visual queries attend over text features."""

    def __init__(self, dim: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.out_dim = dim

    def forward(self, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # MultiheadAttention expects (B, N, D); we have (B, D), so unsqueeze
        q = v.unsqueeze(1)                       # (B, 1, D)
        k = t.unsqueeze(1)                       # (B, 1, D)
        attn_out, _ = self.attn(q, k, k)         # (B, 1, D)
        x = self.norm1(v + attn_out.squeeze(1))  # residual + norm
        x = self.norm2(x + self.ffn(x))          # ffn + residual + norm
        return x                                 # (B, D)


def build_fusion(kind: str = "concat", dim: int = 512, **kwargs) -> nn.Module:
    """Factory. kind in {'concat', 'gated', 'crossattn'}."""
    kind = kind.lower()
    if kind in ("concat", "concat_mlp", "mlp"):
        return ConcatMLPFusion(dim=dim, **kwargs)
    if kind in ("gated", "gate"):
        return GatedFusion(dim=dim, **kwargs)
    if kind in ("crossattn", "cross_attn", "attention"):
        return CrossAttnFusion(dim=dim, **kwargs)
    raise ValueError(f"Unknown fusion kind: {kind}")


if __name__ == "__main__":
    # Smoke test: instantiate all three, run a forward pass, count parameters.
    torch.manual_seed(0)
    B, D = 4, 512
    v = torch.randn(B, D)
    t = torch.randn(B, D)

    print("Fusion modules - smoke test")
    print("=" * 60)
    for kind in ["concat", "gated", "crossattn"]:
        mod = build_fusion(kind, dim=D)
        out = mod(v, t)
        n_params = sum(p.numel() for p in mod.parameters() if p.requires_grad)
        print(f"\n[{kind:>10s}]  out.shape = {tuple(out.shape)}, "
              f"trainable params = {n_params:,}")
        # Verify output isn't identical to either input
        assert not torch.allclose(out, v), f"{kind} output = visual input"
        assert not torch.allclose(out, t), f"{kind} output = text input"
        # Verify no NaNs
        assert not torch.isnan(out).any(), f"{kind} produced NaNs"
        print(f"            output mean = {out.mean().item():+.4f}, "
              f"std = {out.std().item():.4f}")

    print("\n" + "=" * 60)
    print("fusion.py smoke test complete.")
