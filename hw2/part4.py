"""
16-745 Assignment 2 - Part 4: Goal-Conditioned PPO for TWIP Drag Racing
========================================================================
Trains a PPO policy to drive the TWIP to a target wheel position
as fast as possible while staying balanced.

The policy is goal-conditioned:
    obs = [wheel_angle, wheel_velocity, body_angle, body_angular_velocity,
           wheel_angle_goal]

The policy learns to:
  1. Lean forward to accelerate toward goal
  2. Decelerate as it approaches
  3. Arrive balanced with zero velocity

Compares: LQR regulator vs Part 3 PPO vs Part 4 goal-conditioned PPO
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import gym
from gym import spaces

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

TRAIN_MODEL  = False
MODEL_PATH   = "ppo_twip_part4"
VECNORM_PATH = "ppo_twip_part4_vecnorm.pkl"

N_TRAIN_STEPS = 2_000_000   # more steps needed for goal-conditioned task
RENDER_EVERY  = 5

# goal range during training
GOAL_MIN = -3.0   # rad (~1m back)
GOAL_MAX =  3.0   # rad (~1m forward)

# success criteria
GOAL_THRESHOLD    = 0.05   # rad — close enough to goal
VELOCITY_THRESHOLD = 0.1   # rad/s — slow enough to count as stopped

np.random.seed(42)

# ============================================================
# Goal-Conditioned TWIP Environment
# ============================================================

class TWIPGoalEnv(TWIPEnv):
    """
    Extends TWIPEnv with a goal wheel position.
    Observation: [wheel_angle, wheel_velocity, body_angle,
                  body_angular_velocity, wheel_angle_goal]
    Reward: penalizes distance to goal, body tilt, and velocity at goal.
    """

    def __init__(self):
        super().__init__(sim_params_path=SIM_PARAMS_PATH, control_mode="torque")

        # goal sampled each episode
        self.wheel_angle_goal = 0.0

        # extend observation space to include goal
        self.observation_space = spaces.Box(
            low  = np.array([-np.inf, -np.inf, -np.pi, -np.inf, GOAL_MIN],
                            dtype=np.float32),
            high = np.array([ np.inf,  np.inf,  np.pi,  np.inf, GOAL_MAX],
                            dtype=np.float32),
        )

    def reset(self, seed=None, options=None):

        # always start upright at zero
        self.state = np.zeros(4, dtype=np.float32)
        self.step_count = 0
        self.time       = 0.0

        # sample a new goal each episode
        self.wheel_angle_goal = np.random.uniform(GOAL_MIN, GOAL_MAX)

        return self._get_obs(), {}

    def _get_obs(self):
        return np.append(self.state, self.wheel_angle_goal).astype(np.float32)

    def step(self, action):

        torque = float(np.asarray(action).squeeze())

        # fatal error check
        fatal_error = self.is_fatal_error(torque)

        # clip torque for physics
        torque_clipped = np.clip(torque, -self.max_torque, self.max_torque)

        wheel_angle, wheel_velocity, body_angle, body_angular_velocity = self.state

        wheel_accel, body_accel = self.twip3(
            wheel_angle, wheel_velocity,
            body_angle, body_angular_velocity,
            torque_clipped
        )

        # euler integration
        wheel_velocity        += wheel_accel * self.dt
        wheel_angle           += wheel_velocity * self.dt
        body_angular_velocity += body_accel * self.dt
        body_angle            += body_angular_velocity * self.dt

        self.state = np.array(
            [wheel_angle, wheel_velocity, body_angle, body_angular_velocity],
            dtype=np.float32
        )

        self.step_count += 1
        self.time       += self.dt

        # termination conditions
        body_fall  = abs(body_angle) > self.max_body_angle
        truncated  = self.time >= self.max_ep_len
        terminated = body_fall or fatal_error

        # reward
        wheel_error = wheel_angle - self.wheel_angle_goal

        if fatal_error:
            reward = -1e6
        else:
            reward = -(
                1.0  * wheel_error**2              # drive to goal
                + 0.01 * wheel_velocity**2         # arrive with low velocity
                + 10.0 * body_angle**2             # stay upright
                + 0.1  * body_angular_velocity**2  # damp body oscillation
                + 0.001 * torque_clipped**2        # smooth torque
            )

            # bonus for reaching goal balanced
            at_goal     = abs(wheel_error)    < GOAL_THRESHOLD
            nearly_stop = abs(wheel_velocity) < VELOCITY_THRESHOLD
            upright     = abs(body_angle)     < 0.05

            if at_goal and nearly_stop and upright:
                reward += 10.0

        return self._get_obs(), reward, terminated, truncated, {}


# ============================================================
# Train or load
# ============================================================

def get_model():

    if TRAIN_MODEL or not os.path.exists(MODEL_PATH + ".zip"):

        print("Training goal-conditioned PPO...")

        train_env = make_vec_env(lambda: TWIPGoalEnv(), n_envs=8)
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)

        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate = 3e-4,
            gamma         = 0.99,
            n_steps       = 2048,
            batch_size    = 64,
            verbose       = 1,
            policy_kwargs = dict(net_arch=[256, 256])  # larger network for goal conditioning
        )

        model.learn(total_timesteps=N_TRAIN_STEPS)
        model.save(MODEL_PATH)
        train_env.save(VECNORM_PATH)
        print("Model saved.")

        return model, train_env

    else:

        print("Loading goal-conditioned PPO model...")

        eval_env = make_vec_env(lambda: TWIPGoalEnv(), n_envs=1)
        eval_env = VecNormalize.load(VECNORM_PATH, eval_env)
        eval_env.training   = False
        eval_env.norm_reward = False

        model = PPO.load(MODEL_PATH)
        print("Model loaded.")

        return model, eval_env


# ============================================================
# Rollout helpers
# ============================================================

simulator = TWIPEnv(sim_params_path=SIM_PARAMS_PATH, control_mode="torque")
A, B      = simulator.linearize()

Q_lqr = np.diag([0.025, 0.04, 11.0, 0.25])
R_lqr = np.array([[14.7]])
lqr_controller = LQRController(A, B, Q_lqr, R_lqr, simulator.dt)


def run_lqr(goal_wheel_angle, max_steps=2000):
    """
    Run LQR from upright position toward goal.
    LQR treats the error relative to goal as the state.
    """

    simulator.state      = np.zeros(4, dtype=np.float32)
    simulator.time       = 0.0
    simulator.step_count = 0

    logger = Logger()

    for i in range(max_steps):

        t     = simulator.time
        state = simulator.get_state()

        # LQR error relative to goal
        x_err        = state.copy()
        x_err[0]     = state[0] - goal_wheel_angle  # wheel angle error

        u = -lqr_controller.K @ x_err
        u = float(np.clip(u, -simulator.max_torque, simulator.max_torque))

        next_state, reward, done, trunc, _ = simulator.step(u)
        logger.log(t=t, true_state=next_state, control=u)

        if RENDER_EVERY != -1 and i % RENDER_EVERY == 0:
            simulator.render()

        if done or trunc:
            break

    return logger


def run_ppo_goal(model, eval_env, goal_wheel_angle, max_steps=2000):

    simulator.state      = np.zeros(4, dtype=np.float32)
    simulator.time       = 0.0
    simulator.step_count = 0

    logger = Logger()

    for i in range(max_steps):

        t     = simulator.time
        state = simulator.get_state()

        # build goal-conditioned obs
        obs_with_goal = np.append(state, goal_wheel_angle).astype(np.float32)
        obs_norm      = eval_env.normalize_obs(obs_with_goal.reshape(1, -1))

        action, _ = model.predict(obs_norm, deterministic=True)
        u         = float(np.asarray(action).squeeze())

        # use simulator.step() — same as LQR
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


def plot_drag_race(lqr_log, ppo_log, goal, filename):
    """Plot LQR vs goal-conditioned PPO for drag race."""

    lqr_log.to_arrays()
    ppo_log.to_arrays()
    lqr_position  = lqr_log.true_states[:, 0]  * simulator.wheel_radius
    ppo_position  = ppo_log.true_states[:, 0]  * simulator.wheel_radius
    goal_position = goal * simulator.wheel_radius


    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    fig.suptitle(
        f"Drag Race: Drive to wheel angle = {goal:.2f} rad "
        f"({goal * simulator.wheel_radius:.3f} m)",
        fontsize=12, fontweight='bold'
    )

    for idx in range(4):
        row = idx // 2
        col = idx  % 2
        ax  = axes[row, col]

        ax.plot(lqr_log.times, lqr_log.true_states[:, idx],
                'b-', linewidth=1.5, label="LQR")
        ax.plot(ppo_log.times, ppo_log.true_states[:, idx],
                'r--', linewidth=1.5, label="PPO Goal")

        # mark goal on wheel angle plot
        if idx == 0:
            ax.axhline(goal, color='g', linestyle=':', linewidth=1.5,
                       label=f'goal = {goal:.2f}')

        ax.set_ylabel(STATE_LABELS[idx])
        ax.set_xlabel("Time (s)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='k', linewidth=0.6, linestyle='--')

    # position plot
    ax_pos = axes[1, 2]
    ax_pos.plot(lqr_log.times, lqr_position, 'b-', linewidth=1.5, label="LQR")
    ax_pos.plot(ppo_log.times, ppo_position, 'r--', linewidth=1.5, label="PPO Goal")
    ax_pos.axhline(goal_position, color='g', linestyle=':', linewidth=1.5,
                label=f'goal = {goal_position:.3f} m')
    ax_pos.set_ylabel("Position (m)")
    ax_pos.set_xlabel("Time (s)")
    ax_pos.set_title("Position")
    ax_pos.legend(fontsize=8)
    ax_pos.grid(True, alpha=0.3)
    ax_pos.axhline(0, color='k', linewidth=0.6, linestyle='--')
    
    # torque plot
    ax_t = axes[0, 2]
    if len(lqr_log.controls) > 0:
        ax_t.plot(lqr_log.times[:len(lqr_log.controls)],
                  lqr_log.controls, 'b-', linewidth=1.5, label="LQR")
    if len(ppo_log.controls) > 0:
        ax_t.plot(ppo_log.times[:len(ppo_log.controls)],
                  ppo_log.controls, 'r--', linewidth=1.5, label="PPO Goal")

    ax_t.axhline( simulator.max_torque, color='k',
                  linestyle=':', linewidth=1, label='limit')
    ax_t.axhline(-simulator.max_torque, color='k',
                  linestyle=':', linewidth=1)
    ax_t.set_title("Torque (Nm)")
    ax_t.set_xlabel("Time (s)")
    ax_t.set_ylabel("Torque (Nm)")
    ax_t.legend(fontsize=8)
    ax_t.grid(True, alpha=0.3)


    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():

    model, eval_env = get_model()

    # test goals
    goals = [
        (1.0,  "part4_goal_1rad.png"),
        (2.0,  "part4_goal_2rad.png"),
        (-1.5, "part4_goal_neg1.5rad.png"),
    ]

    for goal_wheel_angle, filename in goals:

        print(f"\nDrag race to goal = {goal_wheel_angle:.2f} rad "
              f"({goal_wheel_angle * simulator.wheel_radius:.3f} m)")

        print("  Running LQR...")
        lqr_log = run_lqr(goal_wheel_angle)

        print("  Running goal-conditioned PPO...")
        ppo_log = run_ppo_goal(model, eval_env, goal_wheel_angle)

        plot_drag_race(lqr_log, ppo_log, goal_wheel_angle, filename)

    plt.show()
    print("\nDone.")


if __name__ == "__main__":
    main()