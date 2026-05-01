"""
Live WASD navigation in a Habitat scene.
Opens a window showing the agent's RGB view. Press keys to move in real-time.

Controls:
  W = move forward (0.25 m)
  S = move backward (0.25 m)
  A = turn left  (10 deg)
  D = turn right (10 deg)
  R = reset position
  ESC or Q = quit

NOTE: Click on the window once after it opens so it captures keypresses.
"""
import habitat_sim
import numpy as np
import cv2

SCENE = "data/scene_datasets/habitat-test-scenes/apartment_1.glb"

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
start_state = agent.get_state()

# --- Helpers ---
KEY_TO_ACTION = {
    ord('w'): "move_forward",
    ord('s'): "move_backward",
    ord('a'): "turn_left",
    ord('d'): "turn_right",
    ord('W'): "move_forward",
    ord('S'): "move_backward",
    ord('A'): "turn_left",
    ord('D'): "turn_right",
}

def render():
    """Get the current agent RGB view as a BGR image for OpenCV."""
    obs = sim.get_sensor_observations()
    rgb = obs["rgb"][:, :, :3]              # drop alpha
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)  # OpenCV uses BGR
    return bgr

def overlay_info(img, step, action, pos):
    """Draw status text on top of the image."""
    h, w = img.shape[:2]
    text_lines = [
        f"Step: {step}",
        f"Last: {action}",
        f"Pos: ({pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:+.2f})",
        "WASD=move  R=reset  ESC=quit",
    ]
    for i, line in enumerate(text_lines):
        y = 25 + i * 22
        # black outline for readability
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img

# --- Main loop ---
print("\n" + "=" * 60)
print(f"Scene: {SCENE}")
print("Click the 'Habitat WASD' window to focus, then use WASD to move.")
print("=" * 60 + "\n")

window_name = "Habitat WASD"
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

step = 0
last_action = "(none)"
img = render()
state = agent.get_state()
img = overlay_info(img.copy(), step, last_action, state.position)
cv2.imshow(window_name, img)

while True:
    key = cv2.waitKey(20) & 0xFF   # 20 ms per frame -> ~50 FPS responsive
    if key == 255:                  # no key pressed this frame
        continue

    if key in (27, ord('q'), ord('Q')):   # ESC or Q -> quit
        break

    if key in (ord('r'), ord('R')):       # reset
        agent.set_state(start_state)
        last_action = "reset"
    elif key in KEY_TO_ACTION:
        action = KEY_TO_ACTION[key]
        sim.step(action)
        last_action = action
        step += 1
    else:
        continue   # ignore other keys

    state = agent.get_state()
    img = render()
    img = overlay_info(img.copy(), step, last_action, state.position)
    cv2.imshow(window_name, img)

cv2.destroyAllWindows()
sim.close()
print(f"\nDone. {step} actions taken.")
