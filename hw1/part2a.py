import numpy as np
from scipy.optimize import minimize

from simulator import TWIPEnv


# ============================================================
# Cost parameters
# ============================================================

BODY_ANGLE_WEIGHT = 1.0
BODY_ANGULAR_VELOCITY_WEIGHT = 0.1
FALL_PENALTY = 1000.0
GAIN_REG_WEIGHT = 1e-3


# ============================================================
# Rollout
# ============================================================

def rollout(env, gains, render=False, log=False, reset_options=None):
    """
    Run one episode with fixed feedback gains.

    Args:
        env: TWIPEnv
        gains: [k_wheel_angle, k_wheel_velocity,
                k_body_angle,  k_body_angular_velocity]
        render: whether to render the rollout
        log: whether to log internal states (for plotting)
        reset_options: optional dict passed to env.reset()

    Returns:
        total_cost (float)
    """

    state, _ = env.reset(options=reset_options)
    total_cost = 0.0

    while True:
        state, _, terminated, truncated, _ = env.step(gains, log=log)

        if render:
            env.render()

        # Unpack state explicitly
        wheel_angle = state[0]
        wheel_velocity = state[1]
        body_angle = state[2]
        body_angular_velocity = state[3]

        # Running cost
        step_cost = (
            BODY_ANGLE_WEIGHT * body_angle**2
            + BODY_ANGULAR_VELOCITY_WEIGHT * body_angular_velocity**2
        )
        total_cost += step_cost

        if terminated:
            total_cost += FALL_PENALTY
            break

        if truncated:
            break

    # Gain regularization (keeps optimizer sane)
    total_cost += GAIN_REG_WEIGHT * np.sum(gains**2)

    return total_cost


# ============================================================
# Gain optimization (Nelder–Mead)
# ============================================================

def optimize_gains(env):
    """
    Optimize linear feedback gains using Nelder–Mead.
    """

    def objective(gains):
        return rollout(env, gains, render=False, log=False)

    # Initial guess:
    # [k_wheel_angle, k_wheel_velocity,
    #  k_body_angle,  k_body_angular_velocity]
    initial_gains = np.array(
        [0.0, 0.5, 10.0, 1.0],
        dtype=np.float64,
    )

    result = minimize(
        objective,
        initial_gains,
        method="Nelder-Mead",
        options={
            "maxiter": 10,
            "disp": True,
        },
    )

    return result.x


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Create environment
    # --------------------------------------------------------
    env = TWIPEnv("./sim_params.yaml")
    # env = TWIPEnv()  # uses default parameters

    # --------------------------------------------------------
    # Optimize gains 
    # --------------------------------------------------------
    best_gains = np.round(optimize_gains(env), 3)

    print("\nOptimized gains:")
    print("  k_wheel_angle           =", best_gains[0])
    print("  k_wheel_velocity        =", best_gains[1])
    print("  k_body_angle            =", best_gains[2])
    print("  k_body_angular_velocity =", best_gains[3])

    # --------------------------------------------------------
    # Challenging visual-test initial conditions
    # --------------------------------------------------------

    # Stress-test initialization:
        # Start far from equilibrium with high angular momentum to probe the limits
        # of recoverability under torque constraints. This is a diagnostic scenario.


    visual_reset_options = {
        "x_initial_range": {
            "wh":  [np.deg2rad(180), np.deg2rad(180)],     # wheel rotation (from upright)
            "whd": [np.deg2rad(0), np.deg2rad(0)],       # wheel speed
            "th":  [np.deg2rad(45), np.deg2rad(45)],     # body angle (from upright)
            "thd": [np.deg2rad(100), np.deg2rad(100)],       # body velocity
        }
    }

    # --------------------------------------------------------
    # Visual test (explicit, difficult, interpretable)
    # --------------------------------------------------------
    rollout(
        env,
        best_gains,
        render=True,
        log=True,
        reset_options=visual_reset_options,
    )

    env.plot_logs()
