"""
eval_vln.py - simulator-rollout evaluation for the trained VLN model.

For each val episode:
    1. Reset sim at episode's start pose in its scene
    2. Encode the instruction once with CLIP
    3. Step the model in the simulator (model picks actions, no expert)
    4. Compute success / SPL
    5. Save an MP4 video of the rollout

Run from the habitat-lab folder:
    python code/evaluation/eval_vln.py --ckpt code/results/il_runs/<run>/best.pt
"""
from __future__ import annotations
import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path

import imageio
import numpy as np
import torch

import habitat_sim

# Make local packages importable
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from models.vln_model import VLNModel, NUM_ACTIONS, ACTION_NAMES
from data.instructions import (
    Episode as InstrEpisode, generate_instruction, horizontal_distance,
)


# Constants
ACTION_STOP    = 0
ACTION_FORWARD = 1
ACTION_LEFT    = 2
ACTION_RIGHT   = 3
HABITAT_ACTION_STR = {
    ACTION_FORWARD: "move_forward",
    ACTION_LEFT:    "turn_left",
    ACTION_RIGHT:   "turn_right",
}

SCENE_BASE = Path("data/scene_datasets/habitat-test-scenes")
VAL_DATASET = Path("data/datasets/pointnav/habitat-test-scenes/v1/val/val.json.gz")

IMG_SIZE = 224
MAX_STEPS = 200
SUCCESS_DIST = 0.20


def make_sim(scene_glb: str) -> habitat_sim.Simulator:
    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = scene_glb
    backend.gpu_device_id = 0

    agent = habitat_sim.agent.AgentConfiguration()
    sensor = habitat_sim.CameraSensorSpec()
    sensor.uuid = "rgb"
    sensor.sensor_type = habitat_sim.SensorType.COLOR
    sensor.resolution = [IMG_SIZE, IMG_SIZE]
    sensor.position = [0.0, 1.5, 0.0]
    agent.sensor_specifications = [sensor]
    agent.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)),
        "turn_left":    habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=10.0)),
        "turn_right":   habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=10.0)),
    }
    cfg = habitat_sim.Configuration(backend, [agent])
    return habitat_sim.Simulator(cfg)


def geodesic_distance(sim, a, b) -> float:
    """Shortest path length on the navmesh from a to b, or inf if unreachable."""
    path = habitat_sim.ShortestPath()
    path.requested_start = np.array(a, dtype=np.float32)
    path.requested_end   = np.array(b, dtype=np.float32)
    sim.pathfinder.find_path(path)
    return float(path.geodesic_distance) if path.geodesic_distance > 0 else float("inf")


@torch.no_grad()
def rollout_episode(model: VLNModel, sim, episode: dict, video_path: Path | None):
    """Run the trained model in the simulator for one episode.
    Returns metrics dict + (optionally) writes a video."""
    start_pos = np.array(episode["start_position"], dtype=np.float32)
    start_rot = episode["start_rotation"]
    goal_pos  = np.array(episode["goals"][0]["position"], dtype=np.float32)

    agent = sim.initialize_agent(0)
    state = habitat_sim.AgentState()
    state.position = start_pos
    state.rotation = start_rot
    try:
        agent.set_state(state)
    except Exception:
        pass

    # Optimal path length (denominator of SPL)
    geo_dist = geodesic_distance(sim, start_pos, goal_pos)

    # Instruction
    scene_name = Path(episode["scene_id"]).stem
    instr = generate_instruction(InstrEpisode(
        scene_id=scene_name,
        start_position=tuple(start_pos.tolist()),
        goal_position =tuple(goal_pos.tolist()),
        episode_id=str(episode.get("episode_id", "")),
    ))

    # Encode instruction once
    text_feat = model.encode_text_once([instr])     # (1, 512)
    hidden    = model.init_hidden(1)                # (1, 1, 512)

    # Run the rollout
    frames        = []
    actions_taken = []
    path_length   = 0.0
    prev_pos      = start_pos.copy()
    final_dist    = geo_dist

    NEAR_GOAL_RADIUS = 1.0
    NO_PROGRESS_PATIENCE = 25
    near_goal_count = 0
    best_dist_seen = float('inf')
    no_progress_count = 0

    for step_i in range(MAX_STEPS):
        obs = sim.get_sensor_observations()
        rgb_uint8 = obs["rgb"][:, :, :3].astype(np.uint8)
        frames.append(rgb_uint8.copy())

        # Forward through model
        rgb_tensor = torch.from_numpy(rgb_uint8).unsqueeze(0).float()
        step_out = model.step(rgb_tensor, text_feat, hidden)
        action = int(step_out.action_logits.argmax(dim=-1).item())
        hidden = step_out.hidden

        # Inference-time stop heuristic
        cur_pos = agent.get_state().position
        cur_dist = float(np.linalg.norm(cur_pos - goal_pos))
        if cur_dist < NEAR_GOAL_RADIUS:
            near_goal_count += 1
        else:
            near_goal_count = 0
        if cur_dist < best_dist_seen - 0.05:
            best_dist_seen = cur_dist
            no_progress_count = 0
        else:
            no_progress_count += 1
        if near_goal_count >= 3 and action != ACTION_STOP:
            action = ACTION_STOP
        elif no_progress_count >= NO_PROGRESS_PATIENCE and action != ACTION_STOP:
            action = ACTION_STOP

        actions_taken.append(action)

        if action == ACTION_STOP:
            break

        # Otherwise step the sim
        action_str = HABITAT_ACTION_STR.get(action)
        if action_str is None:
            break
        sim.step(action_str)

        # Update path length (Euclidean between consecutive positions)
        cur_pos = agent.get_state().position
        path_length += float(np.linalg.norm(cur_pos - prev_pos))
        prev_pos = cur_pos

    # Final distance to goal
    final_pos  = agent.get_state().position
    final_dist = float(np.linalg.norm(final_pos - goal_pos))
    success    = bool(final_dist < SUCCESS_DIST)

    # SPL = success * (geo_dist / max(path_length, geo_dist))
    if success and path_length > 0:
        spl = geo_dist / max(path_length, geo_dist)
    else:
        spl = 0.0

    if video_path is not None and len(frames) > 1:
        try:
            imageio.mimsave(str(video_path), frames, fps=10)
        except Exception as e:
            print(f"  (warning: could not save video: {e})")

    return {
        "episode_id":  str(episode.get("episode_id", "")),
        "scene_id":    scene_name,
        "instruction": instr,
        "T_taken":     len(actions_taken),
        "geo_dist":    geo_dist,
        "path_length": path_length,
        "final_dist":  final_dist,
        "success":     success,
        "spl":         float(spl),
        "actions":     actions_taken,
    }


def load_model(ckpt_path: Path) -> VLNModel:
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ckpt.get("args", {})
    fusion = args.get("fusion", "concat")
    model = VLNModel(fusion_kind=fusion, device="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  fusion={fusion}, val_acc at save = {ckpt.get('val_acc', '?')}")
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True,
                   help="Path to a .pt checkpoint (e.g. code/results/il_runs/<run>/best.pt)")
    p.add_argument("--split", type=str, default="val",
                   help="Which split to eval. We use the same val_files as training.")
    p.add_argument("--out-dir", type=str, default=None,
                   help="Where to save videos + summary. Default: alongside the checkpoint.")
    p.add_argument("--max-episodes", type=int, default=None,
                   help="Optional cap (debug).")
    p.add_argument("--no-video", action="store_true")
    args = p.parse_args()

    ckpt_path = Path(args.ckpt)
    out_dir = Path(args.out_dir) if args.out_dir else ckpt_path.parent / "eval_rollout"
    out_dir.mkdir(parents=True, exist_ok=True)
    video_dir = out_dir / "videos"
    if not args.no_video:
        video_dir.mkdir(exist_ok=True)
    print(f"Output dir: {out_dir}")

    # Load val files from training split.json
    split_path = ckpt_path.parent / "split.json"
    if split_path.exists():
        split = json.load(open(split_path))
        val_filenames = split["val_files"]
        # Convert filenames like "scene__epXX.pt" -> {(scene, episode_id)}
        val_keys = set()
        for fn in val_filenames:
            stem = fn.replace(".pt", "")
            scene, _, eid = stem.partition("__ep")
            val_keys.add((scene, eid))
        print(f"  Eval on val split: {len(val_keys)} episodes")
    else:
        print(f"!! split.json not found, evaluating on all val episodes")
        val_keys = None

    # Load val episodes from dataset
    with gzip.open(VAL_DATASET, "rt") as f:
        all_eps = json.load(f).get("episodes", [])

    # Filter to just the val-split episodes from training
    if val_keys is not None:
        episodes = []
        for ep in all_eps:
            scene = Path(ep["scene_id"]).stem
            eid = str(ep.get("episode_id", ""))
            if (scene, eid) in val_keys:
                episodes.append(ep)
        print(f"  Matched {len(episodes)}/{len(val_keys)} requested episodes in dataset")
    else:
        episodes = all_eps

    if args.max_episodes:
        episodes = episodes[: args.max_episodes]

    # Load model
    model = load_model(ckpt_path)

    # Group episodes by scene to avoid creating a new sim per episode
    by_scene = {}
    for ep in episodes:
        by_scene.setdefault(Path(ep["scene_id"]).stem, []).append(ep)

    print(f"\nRunning {len(episodes)} episodes across {len(by_scene)} scene(s)...\n")
    results = []
    t_total = time.time()
    for scene_name, eps in by_scene.items():
        scene_glb = SCENE_BASE / f"{scene_name}.glb"
        if not scene_glb.exists():
            print(f"!! Scene file not found: {scene_glb}, skipping")
            continue
        print(f"---- {scene_name}  ({len(eps)} episodes) ----")
        sim = make_sim(str(scene_glb))
        for ep in eps:
            eid = str(ep.get("episode_id", "?"))
            video_path = None if args.no_video else (
                video_dir / f"{scene_name}__ep{eid}_success={'1' if False else '?'}.mp4")
            t0 = time.time()
            try:
                r = rollout_episode(model, sim, ep, video_path)
            except Exception as e:
                print(f"  ep {eid}: ERROR {e!r}")
                continue
            dt = time.time() - t0
            results.append(r)
            tag = "SUCCESS" if r["success"] else "  fail"
            print(f"  ep {eid}: {tag}  T={r['T_taken']:3d}  "
                  f"final={r['final_dist']:.2f}m  geo={r['geo_dist']:.2f}m  "
                  f"path={r['path_length']:.2f}m  spl={r['spl']:.3f}  ({dt:.1f}s)")

            # Rename video file with success flag for easy filtering
            if video_path is not None and video_path.exists():
                new_name = f"{scene_name}__ep{eid}_success={'1' if r['success'] else '0'}_spl={r['spl']:.2f}.mp4"
                try:
                    video_path.rename(video_dir / new_name)
                except Exception:
                    pass
        sim.close()
        del sim

    # Aggregate
    if not results:
        print("No episodes evaluated.")
        return
    n  = len(results)
    sr = sum(1 for r in results if r["success"]) / n
    spl = sum(r["spl"] for r in results) / n
    avg_dist = sum(r["final_dist"] for r in results) / n

    print("\n" + "=" * 60)
    print("Aggregate metrics")
    print(f"  Episodes evaluated:  {n}")
    print(f"  Success Rate (SR):   {sr:.3f}")
    print(f"  SPL:                 {spl:.3f}")
    print(f"  Avg final distance:  {avg_dist:.2f} m")
    print(f"  Total time:          {time.time()-t_total:.1f}s")

    # Per-scene breakdown
    by_scene_results = {}
    for r in results:
        by_scene_results.setdefault(r["scene_id"], []).append(r)
    print("\nPer-scene breakdown:")
    for scene, rs in by_scene_results.items():
        n_s  = len(rs)
        sr_s = sum(1 for r in rs if r["success"]) / n_s
        spl_s = sum(r["spl"] for r in rs) / n_s
        print(f"  {scene:30s}  n={n_s:3d}  SR={sr_s:.3f}  SPL={spl_s:.3f}")

    # Save summary
    summary = {
        "ckpt":       str(ckpt_path),
        "n_episodes": n,
        "SR":         sr,
        "SPL":        spl,
        "avg_final_dist": avg_dist,
        "per_scene":  {s: {"n": len(rs),
                           "SR": sum(1 for r in rs if r["success"]) / len(rs),
                           "SPL": sum(r["spl"] for r in rs) / len(rs)}
                       for s, rs in by_scene_results.items()},
        "results":    results,
    }
    with open(out_dir / "eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to: {out_dir / 'eval_summary.json'}")
    if not args.no_video:
        print(f"Videos in:          {video_dir}/")


if __name__ == "__main__":
    main()
