# Task 2 — Imitation Learning Training Results

## Setup
- Hardware: Apple M2 Pro, CPU-only PyTorch, M2 GPU for rendering
- Architecture: frozen CLIP-ViT-B/32 visual + text encoders + ConcatMLP fusion + 1-layer GRU + linear policy head
- Trainable parameters: 2,365,444 (fusion: 787K, GRU: 1.5M, policy: 2K)
- Frozen parameters: 151,277,313 (CLIP)
- Training: behavior cloning on 80 expert trajectories, batch size 4, Adam lr 3e-4, 20 epochs
- Validation: 20 held-out trajectories (random split, seed 42)
- Always-forward baseline accuracy: 0.575

## Per-epoch metrics

| Epoch | Train loss | Train acc | Val loss | Val acc |
|-------|-----------|-----------|----------|---------|
| 1     | 1.0753    | 0.574     | 0.9769   | 0.606   |
| 5     | 0.9054    | 0.644     | 0.8902   | 0.642   |
| 10    | 0.8077    | 0.693     | 0.8913   | 0.623   |
| 11    | 0.8182    | 0.683     | 0.8962   | 0.674   |
| 14    | 0.7363    | 0.720     | 0.8296   | 0.669   |
| 19    | 0.6730    | 0.741     | 0.9953   | 0.674   |
| 20    | 0.6511    | 0.742     | 0.9831   | 0.636   |

## Final reported metrics
- **Best val accuracy: 0.674** (at epoch 11 / 19)
- Final train acc: 0.742, final val acc: 0.636
- Train-val gap: 0.106 by epoch 20 (mild overfitting)

## Observations
- Real learning over baseline (+17 train, +10 val accuracy points above 0.575 baseline)
- Mild overfitting from epoch ~14 onward; best.pt checkpoint guards against this
- 80 training trajectories is a small dataset; with more data we expect higher val ceiling
- Action accuracy is per-timestep; navigation success rate (SR/SPL) requires simulator rollout (Step 7)
