# Task 3 - Imitation Learning Training and Evaluation

## 3.1 Training the baseline

The Task 2 vision-language model is trained end-to-end via behavior cloning on pre-rendered GreedyGeodesicFollower expert trajectories. Per-batch, the full padded trajectory passes through the model with teacher forcing; per-step cross-entropy loss against the expert is masked over padding and averaged.

Setup: Adam (lr 3e-4), grad-clip L2 = 1.0, batch size 4, 20 epochs, 80 train / 20 val trajectories from Habitat test scenes, ~3,300 expert action steps, CPU-only PyTorch on Apple M2 Pro, ~110 s/epoch.

The model converges stably: training loss decreases from 1.075 to 0.651, training accuracy rises from the always-forward baseline (0.575) to 0.742, validation accuracy peaks at 0.674 at epoch 11.

## 3.2 Learning curves

Figure 1 (loss) and Figure 2 (accuracy) show clean convergence and mild overfitting from epoch ~14, when training accuracy keeps rising while validation plateaus at ~0.65-0.67. The best.pt checkpoint guards against this by saving the highest-validation-accuracy epoch.

The accuracy curve's distance above the dashed always-forward baseline (0.575) is direct evidence that the model is using vision and language features rather than collapsing to a single action.

Figure 1: code/results/il_runs/concat_lr0.0003_1777577508/learning_curve_loss.png
Figure 2: code/results/il_runs/concat_lr0.0003_1777577508/learning_curve_accuracy.png

## 3.3 Hyperparameter tuning

Two additional runs hold all other settings constant:

| Run                          | LR    | Batch | Epochs | Train acc (final) | Best val acc |
|------------------------------|-------|-------|--------|-------------------|--------------|
| Baseline                     | 3e-4  | 4     | 20     | 0.742             | 0.674        |
| Sweep B (lower LR)           | 1e-4  | 4     | 10     | 0.661             | 0.654        |
| Sweep C (smaller batch)      | 3e-4  | 2     | 10     | 0.689             | 0.659        |

Observations. Both sweeps slightly underperform the baseline best val accuracy (0.674 vs 0.654 / 0.659), but with caveats: Sweep B and C ran 10 epochs vs the baseline's 20, so they are partially under-trained; comparing the baseline at epoch 10 (train 0.693 / val 0.623) is fairer, against which both sweeps actually perform slightly better in val accuracy (0.654 and 0.659 vs 0.623). Lower LR converges more slowly as expected; smaller batch produces more frequent updates and reaches a comparable accuracy in fewer epochs. The comparison confirms lr=3e-4 is a competitive default; smaller batch size is a viable alternative for tighter compute budgets.

## 3.4 Reporting Success Rate (SR) and SPL

The trained model was evaluated by simulator rollout on the 20 validation episodes (9 castle, 11 van-gogh-room): at each timestep the model's predicted action drives the simulator (no expert), with up to 200 steps and a 0.20 m success radius.

Aggregate results:

| Metric                     | Value |
|----------------------------|-------|
| Success Rate (SR)          | 0.000 |
| SPL                        | 0.000 |
| Episodes evaluated         | 20    |
| Mean steps before timeout  | 200   |

Per-scene: SR = 0.000 in both skokloster-castle (n=9) and van-gogh-room (n=11).

Diagnosis - action-distribution collapse:

| Action  | Expert (training) | Trained model (rollout) |
|---------|-------------------|--------------------------|
| stop    | 2.4 %             | 0.0 %                    |
| forward | 57.5 %            | 95.5 %                   |
| left    | 22.9 %            | 3.4 %                    |
| right   | 17.2 %            | 1.1 %                    |

The policy collapsed onto move_forward and never predicts STOP. Without STOP, no episode can succeed (Habitat's PointNav protocol requires the agent to actively call STOP within the goal radius). An inference-time geometric stop heuristic - force STOP if within 1 m of the goal for 3 consecutive steps - also failed to recover non-zero SR, indicating the agent rarely approaches the goal close enough to trigger it.

Why this happens. Textbook behavior cloning under class imbalance: STOP is only 2.4 % of training labels (one per episode), so cross-entropy provides minimal gradient signal for that action. Compounding this is the covariate shift problem (Ross & Bagnell, 2010) - at inference the model's imperfect actions drift the agent into states the expert never visited, where the model's predictions degrade. Per-step accuracy of 0.674 confirms useful visual-language representations are learned at the timestep level, but episode-level navigation fails.

Implications. Class-rebalanced loss (inverse-frequency weighting, available in our training script via --use-class-weights) is the principled fix and is the natural Task 5 controlled extension. Iterative imitation (DAgger) and BC + RL fine-tuning are standard production approaches but exceed the project's compute budget.

## 3.5 Summary

The imitation-learning pipeline trains stably and learns informative multimodal representations (val acc 0.674, +10 pts over the trivial baseline). It fails at episode-level navigation (SR = 0.000) due to action-distribution collapse - a known consequence of behavior cloning under heavy class imbalance. The diagnosis directly motivates the class-weighted variant in Task 5.
