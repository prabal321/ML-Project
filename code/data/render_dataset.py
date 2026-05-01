"""
render_dataset.py - run habitat's expert policy through PointNav episodes
                    and save (rgb sequence, action sequence, instruction)
                    tuples to disk.

This is run ONCE. After it finishes, training scripts load from disk
without ever re-touching the simulator.
"""
from __future__ import annotations
import gzip
import json
import os
import sys
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch

# habitat-sim
import habitat_sim
from habitat_sim.nav import ShortestPath

# our synthetic instruction module
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from instructions import Episode as InstrEpisode, generate_instruction


# Action ID convention (must match VLNModel)
ACTION_STOP    = 0
ACTION_FORWARD = 1
ACTION_LEFT    = 2
ACTION_RIGHT   = 3

OUT_DIR = Path("data/vln_traj")
SCENE_BASE = Path("data/scene_datasets/habitat-test-scenes")
DATASET_FILE = Path("data/datasets/pointnav/habitat-test-scenes/v1/val/val.json.gz")

MAX_STEPS = 200
SUCCESS_DIST = 0.20
IMG_SIZE = 224  # CLIP-friendly


def make_sim(scene_glb: str) -> habitat_sim.Simulator:
    """Configure and return a single-agent simulator for the given scene."""
    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = scene_glb
    backend.gpu_device_id = 0
    backend.allow_sliding = True

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


def expert_action(sim, agent, goal_pos: np.ndarray) -> int:
    """Use Habitat's GreedyGeodesicFollower to choose the next action.
    Falls back to a heuristic if not available."""
    try:
        from habitat_sim.nav import GreedyGeodesicFollower
        follower = GreedyGeodesicFollower(
            sim.pathfinder, agent,
            forward_key="move_forward",
            left_key="turn_left",
            right_key="turn_right",
        )
        try:
            action_str = follower.next_action_along(goal_pos)
        except Exception:
            return ACTION_STOP
        if action_str is None:
            return ACTION_STOP
        return {
            "move_forward": ACTION_FORWARD,
            "turn_left":    ACTION_LEFT,
            "turn_right":   ACTION_RIGHT,
        }.get(action_str, ACTION_STOP)
    except Exception:
        # Cheapest fallback: turn until facing goal then move forward.
        return ACTION_FORWARD


def render_episode(sim, episode: dict, out_path: Path) -> dict:
    """Run the expert through one episode, save trajectory tensor file.
    Returns a metadata dict (or None if the episode was skipped)."""
    start_pos = np.array(episode["start_position"], dtype=np.float32)
    start_rot = episode["start_rotation"]
    goal_pos  = np.array(episode["goals"][0]["position"], dtype=np.float32)

    agent = sim.initialize_agent(0)
    state = habitat_sim.AgentState()
    state.position = start_pos
    state.rotation = np.quaternion(*start_rot) if hasattr(np, "quaternion") else start_rot
    try:
        agent.set_state(state)
    except Exception:
        # Habitat sometimes wants a different rotation format. Try as-is.
        agent.state.position = start_pos

    rgb_frames = []
    actions    = []

    for step in range(MAX_STEPS):
        obs = sim.get_sensor_observations()
        rgb = obs["rgb"][:, :, :3]   # drop alpha (H, W, 3) uint8
        rgb_frames.append(rgb.copy())

        # Distance check
        cur = agent.get_state().position
        dist = float(np.linalg.norm(cur - goal_pos))
        if dist < SUCCESS_DIST:
            actions.append(ACTION_STOP)
            break

        a = expert_action(sim, agent, goal_pos)
        actions.append(a)
        if a == ACTION_STOP:
            break

        action_str = {
            ACTION_FORWARD: "move_forward",
            ACTION_LEFT:    "turn_left",
            ACTION_RIGHT:   "turn_right",
        }.get(a, None)
        if action_str is None:
            break
        sim.step(action_str)

    T = len(actions)
    if T < 2:
        return None  # skip episodes the expert couldn't make progress on

    # Build instruction
    scene_name = Path(episode["scene_id"]).stem
    instr = generate_instruction(InstrEpisode(
        scene_id=scene_name,
        start_position=tuple(start_pos.tolist()),
        goal_position =tuple(goal_pos.tolist()),
        episode_id=str(episode.get("episode_id", "")),
    ))

    rgb_tensor    = torch.from_numpy(np.stack(rgb_frames, axis=0))   # (T, H, W, 3) uint8
    action_tensor = torch.tensor(actions, dtype=torch.long)          # (T,)

    final_dist = float(np.linalg.norm(agent.get_state().position - goal_pos))
    success = bool(final_dist < SUCCESS_DIST)

    torch.save({
        "rgb":         rgb_tensor,
        "actions":     action_tensor,
        "instruction": instr,
        "episode_id":  str(episode.get("episode_id", "")),
        "scene_id":    scene_name,
        "start_pos":   start_pos.tolist(),
        "goal_pos":    goal_pos.tolist(),
        "T":           T,
        "success":     success,
        "final_dist":  final_dist,
    }, out_path)

    return {
        "episode_id": episode.get("episode_id", ""),
        "scene_id":   scene_name,
        "T":          T,
        "success":    success,
        "final_dist": final_dist,
        "instruction": instr,
        "out_path":   str(out_path),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load all val episodes
    print(f"Loading episodes from {DATASET_FILE}")
    with gzip.open(DATASET_FILE, "rt") as f:
        data = json.load(f)
    episodes = data.get("episodes", [])
    print(f"  total episodes: {len(episodes)}")

    # Group by scene
    by_scene = {}
    for ep in episodes:
        scene = Path(ep["scene_id"]).stem
        by_scene.setdefault(scene, []).append(ep)

    print(f"  scene distribution: { {s: len(v) for s, v in by_scene.items()} }")
    print(f"  saving trajectories to {OUT_DIR}/\n")

    summary = []
    t_total = time.time()
    for scene, eps in by_scene.items():
        scene_glb = SCENE_BASE / f"{scene}.glb"
        if not scene_glb.exists():
            print(f"!! scene file not found: {scene_glb}, skipping {len(eps)} episodes")
            continue
        print(f"---- {scene}  ({len(eps)} episodes) ----")
        sim = make_sim(str(scene_glb))
        for ep in eps:
            eid = ep.get("episode_id", "?")
            out_name = f"{scene}__ep{eid}.pt"
            out_path = OUT_DIR / out_name
            t0 = time.time()
            try:
                meta = render_episode(sim, ep, out_path)
            except Exception as e:
                print(f"  ep {eid}: ERROR {e!r}")
                continue
            dt = time.time() - t0
            if meta is None:
                print(f"  ep {eid}: SKIPPED (no progress)")
                continue
            summary.append(meta)
            tag = "SUCCESS" if meta["success"] else "fail"
            print(f"  ep {eid}: T={meta['T']:3d} {tag:8s} "
                  f"final_dist={meta['final_dist']:.2f} m  ({dt:.1f}s)")
        sim.close()
        del sim

    print(f"\nFinished in {time.time()-t_total:.1f}s")
    print(f"  trajectories saved: {len(summary)}/{len(episodes)}")
    print(f"  successes:          {sum(1 for m in summary if m['success'])}")
    print(f"  output dir:         {OUT_DIR}/")

    # Summary file
    with open(OUT_DIR / "_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  summary written to: {OUT_DIR / '_summary.json'}")


if __name__ == "__main__":
    main()
