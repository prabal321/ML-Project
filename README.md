# Vision-Language Reasoning for Visual Navigation

ML course project at IIT Ropar implementing a learning-based vision-
language navigation (VLN) agent in Habitat. Given an RGB observation
and a natural-language instruction, the agent predicts a sequence of
discrete actions (forward, left, right, stop) that drives it to a goal
in a 3-D scene.

## Authors

- Prabal Sharma — 2023CSB1145
- Abhijeet Singh — 2023CSB1094
- Parashdeep Singh — 2023MCB1306

## Quick results

| Metric | Value |
|---|---|
| Per-step val accuracy (baseline) | 0.674 |
| SR / SPL @ rollout (baseline) | 0.000 (action-collapse) |
| Cross-attention extension @ matched 10 ep | +3.9 pp val acc |
| Frozen vs fine-tuned ablation | 0.674 vs 0.648 |
| Paraphrase robustness delta | -2.6 pp |

See `code/results/report/main.tex` for the full report.

## Repository layout
code/
data/           Synthetic instruction generator + dataset/loader
models/         CLIP encoders + fusion variants + VLN model
training/       train_il.py - imitation-learning trainer
evaluation/     Simulator rollout, paraphrase, per-scene evals
results/        Training run outputs (history.json, best.pt, plots)
results/report/ LaTeX report and figures

## Reproducing

This repo contains only the code, the LaTeX report, and a few small
result artifacts. Habitat itself, the Habitat test scenes, and the
pre-rendered expert trajectories are not committed (third-party / too
large / reproducible).

To reproduce from scratch:

1. Install Habitat + dependencies:
```bash
   conda create -n habitat python=3.9 -y
   conda activate habitat
   conda install habitat-sim==0.3.3 -c conda-forge -c aihabitat -y
   pip install habitat-lab==0.3.3 habitat-baselines==0.3.3
   pip install transformers torch imageio matplotlib
```

2. Download the Habitat test scenes:
```bash
   python -m habitat_sim.utils.datasets_download \
     --uids habitat_test_scenes --data-path data/
```

3. Render expert trajectories:
```bash
   python code/data/render_dataset.py
```

4. Train the baseline:
```bash
   python code/training/train_il.py --epochs 20 --lr 3e-4 --batch-size 4
```

5. Evaluate:
```bash
   python code/evaluation/eval_vln.py \
     --ckpt code/results/il_runs/<run-name>/best.pt
```

## Hardware notes

Trained on Apple M2 Pro, CPU-only PyTorch. Each baseline epoch takes
~110s. Mac users may need to set:

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
export HABITAT_ENV_DEBUG=1
```

See report Section 7 ("Implementation Challenges") for a full list of
Mac-specific issues and fixes.

## License

Course-project code, all rights reserved. Habitat-lab and Habitat-sim
(referenced but not included) are MIT-licensed by Meta Platforms Inc.
