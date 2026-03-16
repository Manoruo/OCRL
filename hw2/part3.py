import numpy as np
import matplotlib.pyplot as plt
import os

from utils.simulator import TWIPEnv
from utils.lqr import LQRController
from utils.logger import Logger

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize


SIM_PARAMS_PATH = "./sim_params.yaml"
RENDER = True
RENDER_EVERY = 5

TRAIN_MODEL = False
MODEL_PATH = "ppo_twip_controller"
VECNORM_PATH = "ppo_twip_vecnormalize.pkl"


# ----------------------------------------------------
# Training environment with easier initial conditions
# ----------------------------------------------------

class TrainTWIPEnv(TWIPEnv):
    def __init__(self):
        super().__init__(sim_params_path=SIM_PARAMS_PATH, control_mode="torque")
      




# ----------------------------------------------------
# Evaluation simulator
# ----------------------------------------------------

simulator = TWIPEnv(sim_params_path=SIM_PARAMS_PATH, control_mode="torque")

# ----------------------------------------------------
# LQR controller for comparison
# ----------------------------------------------------

A, B = simulator.linearize()

Q = np.diag([1.0, 1.0, 100.0, 10.0])
R = np.array([[0.1]])

lqr_controller = LQRController(A, B, Q, R, simulator.dt)


# ----------------------------------------------------
# PPO environment
# ----------------------------------------------------

train_env = make_vec_env(lambda: TrainTWIPEnv(), n_envs=4)
train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)


# ----------------------------------------------------
# Train OR Load PPO nonlinear policy
# ----------------------------------------------------

if TRAIN_MODEL or not os.path.exists(MODEL_PATH + ".zip"):

    print("Training PPO nonlinear controller...")

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        gamma=0.99,
        n_steps=2048,
        batch_size=64,
        verbose=1,
    )

    model.learn(total_timesteps=1_000_000)

    model.save(MODEL_PATH)
    train_env.save(VECNORM_PATH)

    print("Model saved")

else:

    print("Loading trained PPO model...")

    train_env = make_vec_env(lambda: TrainTWIPEnv(), n_envs=1)
    train_env = VecNormalize.load(VECNORM_PATH, train_env)

    train_env.training = False
    train_env.norm_reward = False

    model = PPO.load(MODEL_PATH)

    print("Model loaded")


# stop reward normalization during evaluation
train_env.training = False
train_env.norm_reward = False


# ----------------------------------------------------
# Rollout helper
# ----------------------------------------------------

def run_controller(controller_type, initial_state):
    simulator.reset()
    simulator.time = 0.0
    simulator.step_count = 0
    simulator.state = initial_state.copy()

    logger = Logger()

    for i in range(2000):

        t = simulator.time
        state = simulator.get_state()

        if controller_type == "LQR":
            u = lqr_controller.control(state)
        elif controller_type == "PPO":
            obs = train_env.normalize_obs(state.reshape(1, -1))
            action, _ = model.predict(obs, deterministic=True)
            u = np.asarray(action).squeeze()

        else:
            raise ValueError("Unknown controller")

        next_state, reward, done, trunc, _ = simulator.step(u)

        logger.log(
            t=t,
            true_state=next_state,
            control=u,
        )

        if RENDER and i % RENDER_EVERY == 0:
            simulator.render()

        if done or trunc:
            break

    return logger


# ----------------------------------------------------
# Compare from same initial condition
# ----------------------------------------------------

initial_state, _ = simulator.reset()

print("Running LQR controller")
lqr_logger = run_controller("LQR", initial_state)

print("Running PPO controller")
ppo_logger = run_controller("PPO", initial_state)


# ----------------------------------------------------
# Plot comparison
# ----------------------------------------------------

def plot_comparison():

    lqr_logger.to_arrays()
    ppo_logger.to_arrays()

    labels = [
        "wheel_angle",
        "wheel_velocity",
        "body_angle",
        "body_velocity",
    ]

    plt.figure(figsize=(10, 8))

    for i in range(4):

        plt.subplot(4, 1, i + 1)

        plt.plot(
            lqr_logger.times,
            lqr_logger.true_states[:, i],
            label="LQR"
        )

        plt.plot(
            ppo_logger.times,
            ppo_logger.true_states[:, i],
            label="PPO"
        )

        plt.ylabel(labels[i])
        plt.legend()

    plt.xlabel("time (s)")
    plt.tight_layout()
    plt.show(block=True)


plot_comparison()