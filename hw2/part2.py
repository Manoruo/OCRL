"""
16-745 Assignment 2 - Part 2: Kalman Filter for TWIP
=====================================================
Sensors:
  - Wheel encoder    -> wheel angle
  - Accelerometer    -> body angle (via atan2)
  - Gyro             -> body angular velocity

Filter estimates all four states including
wheel velocity (never directly measured).

Experiments:
  - Baseline         : matched Rf and sensor noise
  - High Rf          : filter distrusts sensors
  - Low Rf           : filter trusts sensors
  - Noisy Accel      : accelerometer noisier than encoder/gyro
"""

import numpy as np
import matplotlib.pyplot as plt
from utils.simulator import TWIPEnv
from utils.lqr import LQRController
from utils.kalman import KalmanFilter

# ============================================================
# Config
# ============================================================
np.random.seed(42)
SIM_PARAMS_PATH = "./sim_params.yaml"
RENDER_EVERY    = -1   # set to e.g. 5 to render, -1 to disable

INITIAL_STATE = np.array([
    0.0,   # wheel angle       (rad)
    0.0,   # wheel velocity    (rad/s)
    0.3,   # body angle        (rad) ~17 degrees
    0.0    # body angular vel  (rad/s)
])



# ============================================================
# Sensor model
# ============================================================

def get_measurement(state, gravity, encoder_std, accel_std, gyro_std):
    """
    Simulates noisy sensor measurements.

    Returns y = [wheel_encoder, accel_body_angle, gyro]

    Accelerometer measures body angle via:
        ax = g * sin(body_angle)
        az = g * cos(body_angle)
        body_angle_from_accel = atan2(ax, az)
    """
    wheel_angle           = state[0]
    body_angle            = state[2]
    body_angular_velocity = state[3]

    ax = gravity * np.sin(body_angle)
    az = gravity * np.cos(body_angle)
    body_angle_from_accel = np.arctan2(ax, az)

    y_clean = np.array([
        wheel_angle,
        body_angle_from_accel,
        body_angular_velocity
    ], dtype=float)

    noise = np.array([
        np.random.normal(0, encoder_std),
        np.random.normal(0, accel_std),
        np.random.normal(0, gyro_std)
    ])

    return y_clean + noise


# ============================================================
# Simulation runner
# ============================================================

def run_simulation(simulator, controller, kf, initial_state,
                   encoder_std, accel_std, gyro_std):
    """
    Runs one episode with:
      - process noise added to dynamics
      - measurement noise added to sensors
      - Kalman filter estimating state
      - LQR controller using estimated state

    Returns: times, true_states, estimated_states, torques
    """

    simulator.state      = initial_state.copy().astype(np.float32)
    simulator.step_count = 0
    simulator.time       = 0.0

    times       = []
    true_states = []
    est_states  = []
    torques     = []

    max_steps = int(simulator.max_ep_len / simulator.dt)
    u = 0.0

    for i in range(max_steps):

        state = simulator.state.copy()
        t     = simulator.time

        # noisy measurement
        y = get_measurement(
            state, simulator.gravity,
            encoder_std, accel_std, gyro_std
        )

        # kalman filter: predict + update
        x_hat = kf.step(u, y)

        # LQR control on estimated state
        u              = controller.control(x_hat)
        torque_clipped = np.clip(u, -simulator.max_torque, simulator.max_torque)

        # render
        if RENDER_EVERY != -1 and i % RENDER_EVERY == 0:
            simulator.render()

        # log before stepping
        times.append(t)
        true_states.append(state.copy())
        est_states.append(x_hat.copy())
        torques.append(u)

        # integrate dynamics
        wh, whd, th, thd = simulator.state
        whdd, thdd = simulator.twip3(wh, whd, th, thd, torque_clipped)

        new_whd = whd + whdd * simulator.dt
        new_wh  = wh  + new_whd * simulator.dt
        new_thd = thd + thdd * simulator.dt
        new_th  = th  + new_thd * simulator.dt

        # add process noise to simulate real world disturbances
    
        simulator.state = np.array(
            [new_wh, new_whd, new_th, new_thd], dtype=np.float32
        ) 

        simulator.time       += simulator.dt
        simulator.step_count += 1

        # stop if fallen or diverged
        if abs(new_th) > simulator.max_body_angle or not np.isfinite(new_th):
            print(f"  [!] Fell over at t={simulator.time:.2f}s")
            break

    return (
        np.array(times),
        np.array(true_states),
        np.array(est_states),
        np.array(torques)
    )


# ============================================================
# Plotting
# ============================================================

STATE_LABELS = [
    "Wheel Angle (rad)",
    "Wheel Velocity (rad/s)",
    "Body Angle (rad)",
    "Body Ang. Velocity (rad/s)"
]

def plot_results(times, true_states, est_states, title, filename):
    """True vs estimated for all four states."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.suptitle(title, fontsize=12, fontweight='bold')
    axes = axes.flatten()

    for idx, (ax, label) in enumerate(zip(axes, STATE_LABELS)):
        ax.plot(times, true_states[:, idx], 'b-',
                linewidth=1.5, label='True')
        ax.plot(times, est_states[:, idx],  'r--',
                linewidth=1.5, label='Estimated')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='k', linewidth=0.6, linestyle='--')

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  Saved: {filename}")
    plt.close(fig)


def plot_comparison(all_results, filename):
    """Estimation error across all experiments."""

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("Kalman Filter: Estimation Error Comparison",
                 fontsize=12, fontweight='bold')

    colors = ['blue', 'red', 'green', 'orange']

    for (name, times, true_states, est_states), color in zip(all_results, colors):
        body_angle_error = true_states[:, 2] - est_states[:, 2]
        wheel_vel_error  = true_states[:, 1] - est_states[:, 1]

        axes[0].plot(times, body_angle_error,
                     color=color, label=name, linewidth=1.5)
        axes[1].plot(times, wheel_vel_error,
                     color=color, label=name, linewidth=1.5)

    axes[0].set_title("Body Angle Error")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Error (rad)")
    axes[0].axhline(0, color='k', linewidth=0.6, linestyle='--')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Wheel Velocity Error (unobserved state)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Error (rad/s)")
    axes[1].axhline(0, color='k', linewidth=0.6, linestyle='--')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  Saved: {filename}")
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():

    # ----------------------------------
    # Setup simulator and linearization
    # ----------------------------------
    simulator = TWIPEnv(sim_params_path=SIM_PARAMS_PATH, control_mode="torque")
    A, B      = simulator.linearize()
    dt        = simulator.dt

    # discrete dynamics for Kalman filter
    Ad = np.eye(4) + A * dt
    Bd = B * dt

    # measurement matrix
    # y = [wheel_encoder, accel_body_angle, gyro]
    # C selects [wheel_angle, body_angle, body_angular_velocity] from state
    C = np.array([
        [1, 0, 0, 0],   # wheel encoder    -> wheel angle
        [0, 0, 1, 0],   # accelerometer    -> body angle
        [0, 0, 0, 1]    # gyro             -> body angular velocity
    ])

    # LQR controller (Bryson scaled from part 1)
    Q_lqr = np.diag([0.025, 0.04, 11.0, 0.25])
    R_lqr = np.array([[14.7]])
    controller = LQRController(A, B, Q_lqr, R_lqr, dt)

    # process noise covariance — fixed across all experiments
    # represents model uncertainty (linearization error, disturbances)
    # process noise covariance — design variable, fixed for all experiments
    Qf = np.diag([
        1e-3,   # wheel angle
        1e-2,   # wheel velocity
        1e-3,   # body angle
        1e-2    # body angular velocity
    ])

    experiments = [
        (
            "Baseline",
            np.diag([
                1e-2,   # encoder variance
                1e-2,   # accelerometer variance
                1e-2,   # gyro variance
            ]),
            dict(encoder_std=0.01, accel_std=0.01, gyro_std=0.01),
            "Balanced Rf — matched to sensor noise"
        ),
        (
            "High Rf",
            np.diag([
                5e-2,   # encoder variance — distrusted
                5e-2,   # accelerometer variance — distrusted
                5e-2,   # gyro variance — distrusted
            ]),
            dict(encoder_std=0.01, accel_std=0.01, gyro_std=0.01),
            "High Rf — filter relies more on model"
        ),
        (
            "Low Rf",
            np.diag([
                1e-4,   # encoder variance — trusted
                1e-4,   # accelerometer variance — trusted
                1e-4,   # gyro variance — trusted
            ]),
            dict(encoder_std=0.01, accel_std=0.01, gyro_std=0.01),
            "Low Rf — filter relies more on sensors"
        ),
        (
            "Noisy Accelerometer",
            np.diag([
                1e-2,   # encoder variance — normal
                0.02,   # accelerometer variance — less trusted
                1e-2,   # gyro variance — normal
            ]),
            dict(encoder_std=0.01, accel_std=0.02, gyro_std=0.01),
            "Accelerometer variance higher than encoder"
        ),
    ]

    # ----------------------------------
    # Run experiments
    # ----------------------------------
    all_results = []

    print("=" * 60)
    print("Part 2: Kalman Filter Experiments")
    print("=" * 60)
    print(f"\nC =\n{C}")
    print(f"\nQf diag           = {np.diag(Qf)}")

    for name, Rf, sensor_noise, desc in experiments:

        encoder_std = sensor_noise["encoder_std"]
        accel_std   = sensor_noise["accel_std"]
        gyro_std    = sensor_noise["gyro_std"]

        print(f"\n[{name}]")
        print(f"  {desc}")
        print(f"  Rf diag     = {np.diag(Rf)}")
        print(f"  encoder_std = {encoder_std}")
        print(f"  accel_std   = {accel_std}")
        print(f"  gyro_std    = {gyro_std}")

        # initialize filter at true starting state
        kf       = KalmanFilter(Ad, Bd, C, Qf, Rf)
        kf.x_hat = INITIAL_STATE.copy()

        times, true_states, est_states, torques = run_simulation(
            simulator, controller, kf, INITIAL_STATE,
            encoder_std, accel_std, gyro_std
        )

        rms_body_angle = np.sqrt(np.mean((true_states[:, 2] - est_states[:, 2])**2))
        rms_wheel_vel  = np.sqrt(np.mean((true_states[:, 1] - est_states[:, 1])**2))

        print(f"  RMS body angle error  = {rms_body_angle:.5f} rad")
        print(f"  RMS wheel vel error   = {rms_wheel_vel:.5f} rad/s")

        plot_results(
            times, true_states, est_states,
            title=f"Kalman Filter: {name} — {desc}",
            filename=f"part2_{name.replace(' ', '_').lower()}.png"
        )

        all_results.append((name, times, true_states, est_states))

    plot_comparison(all_results, filename="part2_comparison.png")

    plt.show()
    print("\nDone.")


if __name__ == "__main__":
    main()