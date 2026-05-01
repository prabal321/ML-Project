"""
Step-by-step WASD navigation in a Habitat scene.
Type a key + Enter to take an action. Each step saves the agent's view as a PNG.

Controls:
  w = move forward (0.25 m)
  s = move backward (0.25 m)
  a = turn left (10 deg)
  d = turn right (10 deg)
  q = quit
"""
import habitat_sim
from PIL import Image
import os

SCENE = "data/scene_datasets/habitat-test-scenes/apartment_1.glb"
OUT_DIR = "wasd_views"
os.makedirs(OUT_DIR, exist_ok=True)

# --- Simulator setup ---
backend_cfg = habitat_sim.SimulatorConfiguration()
backend_cfg.scene_id = SCENE

agent_cfg = habitat_sim.agent.AgentConfiguration()
sensor = habitat_sim.CameraSensorSpec()
sensor.uuid = "rgb"
sensor.sensor_type = habitat_sim.SensorType.COLOR
sensor.resolution = [512, 512]
sensor.position = [0.0, 1.5, 0.0]
agent_cfg.sensor_specifications = [sensor]

agent_cfg.action_space = {
    "move_forward":  habitat_sim.agent.ActionSpec("move_forward",  habitat_sim.agent.ActuationSpec(amount=0.25)),
    "move_backward": habitat_sim.agent.ActionSpec("move_backward", habitat_sim.agent.ActuationSpec(amount=0.25)),
    "turn_left":     habitat_sim.agent.ActionSpec("turn_left",     habitat_sim.agent.ActuationSpec(amount=10.0)),
    "turn_right":    habitat_sim.agent.ActionSpec("turn_right",    habitat_sim.agent.ActuationSpec(amount=10.0)),
}

cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
sim = habitat_sim.Simulator(cfg)
agent = sim.initialize_agent(0)

KEY_TO_ACTION = {"w": "move_forward", "s": "move_backward",
                 "a": "turn_left",    "d": "turn_right"}

def save_view(step_idx):
    """Save current agent view to disk."""
    obs = sim.get_sensor_observations()
    rgb = obs["rgb"][:, :, :3]  # drop alpha channel
    path = os.path.join(OUT_DIR, f"step_{step_idx:03d}.png")
    Image.fromarray(rgb).save(path)
    return path

# Save the starting view
print("\n" + "="*60)
print(f"Scene: {SCENE}")
print("Controls: w=forward, s=back, a=left, d=right, q=quit")
print(f"Views are saved to: ./{OUT_DIR}/")
print("Tip: open the folder in Finder and watch images appear.")
print("="*60)

step = 0
path = save_view(step)
state = agent.get_state()
print(f"\nStart  | pos=({state.position[0]:+.2f}, {state.position[1]:+.2f}, {state.position[2]:+.2f}) | saved {path}")

while True:
    key = input("\nAction (w/a/s/d/q): ").strip().lower()
    if key == "q":
        break
    if key not in KEY_TO_ACTION:
        print(f"Unknown key '{key}'. Use w/a/s/d/q.")
        continue
    action = KEY_TO_ACTION[key]
    sim.step(action)
    step += 1
    state = agent.get_state()
    path = save_view(step)
    print(f"Step {step:3d} | action={action:14s} | "
          f"pos=({state.position[0]:+.2f}, {state.position[1]:+.2f}, {state.position[2]:+.2f}) | saved {path}")

sim.close()
print(f"\nDone. {step} actions taken. Views saved in ./{OUT_DIR}/")
