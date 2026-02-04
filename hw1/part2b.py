import numpy as np
from stable_baselines3 import PPO
from simulator import TWIPEnv


# ============================================================
# Main: PPO Training + Test (Part 2b)
# ============================================================

if __name__ == "__main__":

    params = {
        "m_w": 0.173,
        "m_p": 0.826 - 0.173,
        "I_w": 0.0066,
        "l_p": 0.043,
        "I_p": 0.00084,
        "r_w": 0.0323,
        "g": 9.81,
        "f": 0.0,
        "dt": 1 / 333,
        "x_initial_range": {
            "wh": [-1.0 / 0.0323, 1.0 / 0.0323],
            "whd": [-20, 20],
            "th": [-np.pi / 4, np.pi / 4],
            "thd": [-20, 20],
        },
        "max_torque": 5.0,
        "ep_len": 3 * 333,
    }

    env = TWIPEnv(params)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        verbose=1,
        tensorboard_log="./ppo_twip_logs/"
    )


    model.learn(total_timesteps=200_000)
    print("Training complete")
    # --------------------
    # Test trained policy
    # --------------------
    obs, _ = env.reset()

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action, log=True)
        env.render()
        if terminated or truncated:
            env.plot_logs()
            break
