import habitat_sim
import numpy as np

# Configure the simulator
backend_cfg = habitat_sim.SimulatorConfiguration()
backend_cfg.scene_id = "data/scene_datasets/habitat-test-scenes/apartment_1.glb"

# Configure an agent with an RGB camera sensor
agent_cfg = habitat_sim.agent.AgentConfiguration()
rgb_sensor_spec = habitat_sim.CameraSensorSpec()
rgb_sensor_spec.uuid = "rgb"
rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
rgb_sensor_spec.resolution = [256, 256]
rgb_sensor_spec.position = [0.0, 1.5, 0.0]
agent_cfg.sensor_specifications = [rgb_sensor_spec]

# Build the simulator
cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
sim = habitat_sim.Simulator(cfg)

# Take one observation
obs = sim.get_sensor_observations()
rgb = obs["rgb"]

print("Scene loaded successfully")
print("RGB observation shape:", rgb.shape)
print("RGB dtype:", rgb.dtype)
print("Min/Max pixel values:", rgb.min(), "/", rgb.max())

sim.close()
print("\n✓ habitat-sim is working end-to-end!")
