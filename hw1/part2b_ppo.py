import os
import numpy as np
import torch
from stable_baselines3 import PPO
from simulator import TWIPEnv
import gymnasium as gym
from gymnasium import spaces

# ============================================================
# Environment Wrapper for SB3
# ============================================================
class SB3TWIPWrapper(gym.Env):
    def __init__(self, config_path):
        super().__init__()
        self.twip = TWIPEnv(config_path)
        
        # Action: 4 Gains scaled [-1, 1]
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        # Observation: [x, x_dot, theta, theta_dot]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

    def _rescale_action(self, action):
        # Maps RL output [-1, 1] to physical gain ranges (e.g., 0 to 20)
        return (action + 1.0) * 10.0 

    def step(self, action, log=False):
        gains = self._rescale_action(action)
        # Step the underlying simulator
        state, reward, terminated, truncated, _ = self.twip.step(gains, log=log)
        
        return state.astype(np.float32), reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        state, info = self.twip.reset(options=options)
        return state.astype(np.float32), info

    def render(self):
        self.twip.render()

# ============================================================
# Main Execution
# ============================================================
MODEL_PATH = "ppo_twip_gains"
SIM_PARAMS = "./sim_params.yaml"

env = SB3TWIPWrapper(SIM_PARAMS)

if os.path.exists(f"{MODEL_PATH}.zip"):
    print(f"Loading existing model: {MODEL_PATH}")
    model = PPO.load(MODEL_PATH, env=env)
else:
    print("No model found. Starting training...")
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=1e-3)
    model.learn(total_timesteps=500_000)
    model.save(MODEL_PATH)
    print("Training complete and model saved.")

# ============================================================
# Visualization / Testing
# ============================================================
print("Running simulation with learned policy...")
obs, _ = env.reset()
for _ in range(1000):
    # predict() returns (action, next_state)
    action, _ = model.predict(obs, deterministic=True)
    print(action)
    obs, reward, terminated, truncated, _ = env.step(action, log=True)
    
    env.render()
    
    if terminated or truncated:
        env.twip.plot_logs()