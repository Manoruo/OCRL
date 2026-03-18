"""
16-745 Assignment 2 - Part 3: Nonlinear Policy for TWIP
========================================================
Trains a PPO policy to balance the TWIP accounting for:
  - Full nonlinear dynamics (sin/cos terms)
  - Motor torque limit (fatal if exceeded)
  - Wheel slip constraint (fatal if exceeded)

Compares PPO vs LQR from multiple initial conditions
to identify where nonlinear policy outperforms linear.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from utils.simulator import TWIPEnv
from utils.lqr import LQRController
from utils.logger import Logger

# ============================================================
# Config
# ============================================================

SIM_PARAMS_PATH = "./sim_params.yaml"

TRAIN_MODEL  = False          # set True to retrain
MODEL_PATH   = "ppo_twip_controller"
VECNORM_PATH = "ppo_twip_vecnormalize.pkl"

RENDER_EVERY = -1           # set to e.g. 5 to render, -1 to disable
N_TRAIN_STEPS = 1_000_000

# fixed seed for reproducible comparison
np.random.seed(42)

# ============================================================
# Setup simulator and LQR
# ============================================================

simulator = TWIPEnv(sim_params_path=SIM_PARAMS_PATH, control_mode="torque")
A, B      = simulator.linearize()

# LQR — Bryson scaled from part 1
Q_lqr = np.diag([0.025, 0.04, 11.0, 0.25])
R_lqr = np.array([[14.7]])
lqr_controller = LQRController(A, B, Q_lqr, R_lqr, simulator.dt)

# ============================================================
# Train or load PPO
# ============================================================

train_env = make_vec_env(
    lambda: TWIPEnv(sim_params_path=SIM_PARAMS_PATH, control_mode="torque"),
    n_envs=4
)
train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)

if TRAIN_MODEL or not os.path.exists(MODEL_PATH + ".zip"):

    print("Training PPO nonlinear controller...")

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate = 3e-4,
        gamma         = 0.99,
        n_steps       = 2048,
        batch_size    = 64,
        verbose       = 1,
    )

    model.learn(total_timesteps=N_TRAIN_STEPS)
    model.save(MODEL_PATH)
    train_env.save(VECNORM_PATH)
    print("Model saved.")

else:

    print("Loading trained PPO model...")

    train_env = make_vec_env(
        lambda: TWIPEnv(sim_params_path=SIM_PARAMS_PATH, control_mode="torque"),
        n_envs=1
    )
    train_env = VecNormalize.load(VECNORM_PATH, train_env)
    model     = PPO.load(MODEL_PATH)
    print("Model loaded.")

# disable normalization during evaluation
train_env.training   = False
train_env.norm_reward = False

# ============================================================
# Rollout helper
# ============================================================

def run_episode(controller_type, initial_state, max_steps=2000):
    """
    Run one episode with either LQR or PPO controller.
    Returns a Logger with logged states and controls.
    Torque is clipped for LQR (no fatal error check) so
    we can see full LQR behavior for comparison.
    """

    simulator.state      = initial_state.copy().astype(np.float32)
    simulator.time       = 0.0
    simulator.step_count = 0

    logger = Logger()
    print(f"Running episode for {controller_type}")
    for i in range(max_steps):

        t     = simulator.time
        state = simulator.get_state()

        if controller_type == "LQR":
            u = lqr_controller.control(state)
            # clip torque for LQR — no fatal error check in part 3
            u = np.clip(u, -simulator.max_torque, simulator.max_torque)

        elif controller_type == "PPO":
            obs            = train_env.normalize_obs(state.reshape(1, -1))
            action, _      = model.predict(obs, deterministic=True)
            u              = float(np.asarray(action).squeeze())

        else:
            raise ValueError(f"Unknown controller: {controller_type}")

        # step simulator
        next_state, reward, done, trunc, _ = simulator.step(u)

        logger.log(t=t, true_state=next_state, control=u)

        if RENDER_EVERY != -1 and i % RENDER_EVERY == 0:
            simulator.render()

        if done or trunc:
            break

    return logger

# ============================================================
# Plotting
# ============================================================

STATE_LABELS = [
    "Wheel Angle (rad)",
    "Wheel Velocity (rad/s)",
    "Body Angle (rad)",
    "Body Ang. Velocity (rad/s)"
]

def plot_comparison(lqr_logger, ppo_logger, initial_state, filename):
    """Plot LQR vs PPO states and torque side by side."""

    lqr_logger.to_arrays()
    ppo_logger.to_arrays()

    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    fig.suptitle(
        f"LQR vs PPO  |  Initial body angle = {np.degrees(initial_state[2]):.1f}°",
        fontsize=12, fontweight='bold'
    )

    colors = {"LQR": "blue", "PPO": "red"}

    # state plots
    for idx in range(4):
        row = idx // 2
        col = idx  % 2
        ax  = axes[row, col]

        ax.plot(lqr_logger.times, lqr_logger.true_states[:, idx],
                color=colors["LQR"], linewidth=1.5, label="LQR")
        ax.plot(ppo_logger.times, ppo_logger.true_states[:, idx],
                color=colors["PPO"], linewidth=1.5, label="PPO", linestyle='--')

        ax.set_ylabel(STATE_LABELS[idx])
        ax.set_xlabel("Time (s)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='k', linewidth=0.6, linestyle='--')

    # torque plot
    ax_torque = axes[0, 2]
    if len(lqr_logger.controls) > 0:
        ax_torque.plot(
            lqr_logger.times[:len(lqr_logger.controls)],
            lqr_logger.controls,
            color=colors["LQR"], linewidth=1.5, label="LQR"
        )
    if len(ppo_logger.controls) > 0:
        ax_torque.plot(
            ppo_logger.times[:len(ppo_logger.controls)],
            ppo_logger.controls,
            color=colors["PPO"], linewidth=1.5, label="PPO", linestyle='--'
        )

    ax_torque.axhline( simulator.max_torque, color='k',
                       linestyle=':', linewidth=1, label='torque limit')
    ax_torque.axhline(-simulator.max_torque, color='k',
                       linestyle=':', linewidth=1)
    ax_torque.set_title("Torque (Nm)")
    ax_torque.set_xlabel("Time (s)")
    ax_torque.set_ylabel("Torque (Nm)")
    ax_torque.legend(fontsize=8)
    ax_torque.grid(True, alpha=0.3)

    # hide unused subplot
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close(fig)


# ============================================================
# Main comparison
# ============================================================

def main():

    # ----------------------------------
    # Test 1 — small tilt (LQR should work fine)
    # ----------------------------------
    initial_small = np.array([0.0, 0.0, 0.15, 0.0])

    print("\nTest 1: Small tilt (0.15 rad / ~9 degrees)")
    lqr_log = run_episode("LQR", initial_small)
    ppo_log = run_episode("PPO", initial_small)
    plot_comparison(lqr_log, ppo_log, initial_small,
                    filename="part3_small_tilt.png")

    # ----------------------------------
    # Test 2 — large tilt (nonlinear should help)
    # ----------------------------------
    initial_large = np.array([0.0, 0.0, 0.6, 0.0])

    print("\nTest 2: Large tilt (0.6 rad / ~34 degrees)")
    lqr_log = run_episode("LQR", initial_large)
    ppo_log = run_episode("PPO", initial_large)
    plot_comparison(lqr_log, ppo_log, initial_large,
                    filename="part3_large_tilt.png")

    # ----------------------------------
    # Test 3 — large tilt + angular velocity
    # ----------------------------------
    initial_aggressive = np.array([0.0, 0.0, 0.5, 1.5])

    print("\nTest 3: Large tilt + angular velocity (0.5 rad, 1.5 rad/s)")
    lqr_log = run_episode("LQR", initial_aggressive)
    ppo_log = run_episode("PPO", initial_aggressive)
    plot_comparison(lqr_log, ppo_log, initial_aggressive,
                    filename="part3_aggressive.png")

    plt.show()
    print("\nDone.")


if __name__ == "__main__":
    main()