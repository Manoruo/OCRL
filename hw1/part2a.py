import numpy as np
from scipy.optimize import minimize

from simulator import TWIPEnv, linear_feedback_controller


# ============================================================
# Rollout evaluation
# ============================================================

def evaluate_gains(gains, env, max_steps=1000):
    """
    Runs one rollout with fixed gains and returns a scalar cost.
    Lower cost = better.
    """

    state, _ = env.reset()
    total_cost = 0.0

    for _ in range(max_steps):
        torque = linear_feedback_controller(state, gains)

        state, _, terminated, truncated, _ = env.step([torque])
        done = terminated or truncated

        body_angle = state[2]
        body_angular_velocity = state[3]

        # Penalize deviation from upright
        total_cost += body_angle**2 + 0.1 * body_angular_velocity**2

        if done:
            # Heavy penalty for falling
            total_cost += 1000.0
            break

    # Soft regularization to avoid insane gains
    total_cost += 1e-3 * np.sum(gains**2)

    return total_cost


# ============================================================
# Gain optimization (Nelder–Mead)
# ============================================================

def optimize_gains(env):
    def objective(gains):
        return evaluate_gains(gains, env)

    # Initial guess (does NOT need to be good)
    x0 = np.array([0.0, 0.5, 10.0, 1.0])

    result = minimize(
        objective,
        x0,
        method="Nelder-Mead",
        options={
            "maxiter": 300,
            "disp": True,
        },
    )

    return result.x


# ============================================================
# Main
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
            "whd": [0, 0],
            "th": [-np.pi / 4, np.pi / 4],
            "thd": [0, 0],
        },
        "max_torque": 5.0,
        "ep_len": 3 * 333,
    }

    env = TWIPEnv(params)

    best_gains = optimize_gains(env)

    print("\nOptimized gains:")
    print(best_gains)

    # --------------------------------------------------------
    # Test optimized controller visually
    # --------------------------------------------------------

    state, _ = env.reset()

    while True:
        torque = linear_feedback_controller(state, best_gains)
        state, _, terminated, truncated, _ = env.step([torque])
        env.render()
        if terminated or truncated:
            break
