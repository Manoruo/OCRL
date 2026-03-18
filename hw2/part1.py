"""
16-745 Assignment 2 - Part 1: LQR Controller for TWIP
======================================================
Demonstrates:
  - Linearization of nonlinear TWIP dynamics
  - LQR design with Q and R cost matrices
  - Four tuning experiments comparing controller behavior
  - State plots for writeup
"""

import numpy as np
import matplotlib.pyplot as plt
from utils.simulator import TWIPEnv
from utils.lqr import LQRController

# ============================================================
# Config
# ============================================================

SIM_PARAMS_PATH = "./sim_params.yaml"
INITIAL_BODY_ANGLE = np.pi/6  #
RENDER_EVERY = 5 # -1 to not render
# Fixed initial condition for all experiments (not random)
INITIAL_STATE = np.array([
    0.0,   # wheel angle
    0.0,   # wheel angular velocity  
    INITIAL_BODY_ANGLE,   # body angle (~17 degrees, already a meaningful tilt)
    0    # body angular velocity (already falling forward)
])


# ============================================================
# Simulation runner
# ============================================================

def run_simulation(simulator, controller, initial_state, ep_len):
    """
    Run one simulation episode.
    Returns arrays of time, states, and torques.
    Disables fatal error termination for LQR testing.
    """

    # Override initial state directly
    simulator.state     = initial_state.copy().astype(np.float32)
    simulator.step_count = 0
    simulator.time       = 0.0

    times, states, torques = [], [], []

    max_steps = int(ep_len / simulator.dt)

    for i in range(max_steps):
        state  = simulator.state.copy()
        torque = controller.control(state)

    
        times.append(simulator.time)
        states.append(state.copy())
        torques.append(torque)

        # Step simulator using clipped torque
        torque_clipped = np.clip(torque, -simulator.max_torque, simulator.max_torque)
        # Temporarily bypass fatal error check by calling dynamics directly
        wh, whd, th, thd = simulator.state
        whdd, thdd = simulator.twip3(wh, whd, th, thd, torque_clipped)

        new_whd = whd + whdd * simulator.dt
        new_wh  = wh  + new_whd * simulator.dt
        new_thd = thd + thdd * simulator.dt
        new_th  = th  + new_thd * simulator.dt

        simulator.state = np.array([new_wh, new_whd, new_th, new_thd], dtype=np.float32)
        simulator.time += simulator.dt
        simulator.step_count += 1

        if RENDER_EVERY != -1 and i % RENDER_EVERY == 0:
            simulator.render()

        # Stop if body falls over
        if abs(new_th) > simulator.max_body_angle:
            print(f"  [!] Body fell over at t={simulator.time:.2f}s")
            break

    return (
        np.array(times),
        np.array(states),
        np.array(torques)
    )


# ============================================================
# Plotting
# ============================================================

def plot_experiment(ax_row, times, states, torques, label, color):
    """Plot one experiment onto a row of axes."""

    ax_row[0].plot(times, states[:, 0], color=color, label=label)
    ax_row[1].plot(times, states[:, 1], color=color, label=label)
    ax_row[2].plot(times, states[:, 2], color=color, label=label)
    ax_row[3].plot(times, states[:, 3], color=color, label=label)
    ax_row[4].plot(times, torques,       color=color, label=label)


def make_figure(title):
    """Create a figure with 5 subplots for the 4 states + torque."""

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    fig.suptitle(title, fontsize=13, fontweight='bold')

    labels = [
        "Wheel Angle (rad)",
        "Wheel Velocity (rad/s)",
        "Body Angle (rad)",
        "Body Angular Velocity (rad/s)",
        "Torque (Nm)"
    ]

    for ax, lbl in zip(axes, labels):
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(lbl)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='k', linewidth=0.8, linestyle='--')

    return fig, axes


# ============================================================
# Main
# ============================================================

def main():

    # ----------------------------------
    # Step 1: Linearize
    # ----------------------------------
    simulator = TWIPEnv(sim_params_path=SIM_PARAMS_PATH, control_mode="torque")
    A, B = simulator.linearize()

    print("=" * 60)
    print("TWIP Linearization at Equilibrium")
    print("=" * 60)
    print(f"\nMax torque: {simulator.max_torque:.4f} Nm")
    print(f"Inital body angle: {np.rad2deg(INITIAL_BODY_ANGLE)}")
    print(f"dt:         {simulator.dt:.6f} s")
    print(f"\nA =\n{np.round(A, 3)}")
    print(f"\nB =\n{np.round(B, 3)}")
    print()

    # ----------------------------------
    # Step 2: Define experiments
    # ----------------------------------
    # Each entry: (name, Q, R, color, description)
    experiments = [
        (
            "Balanced",
            np.diag([0.025, 0.04, 11.0, 0.25]),
            np.array([[14.7]]),
            "blue",
            "Balanced Q and R"
        ),
        (
            "Gentle",
            np.diag([0.025, 0.04, 11.0, 0.25]),
            np.array([[150.0]]),
            "red",
            "High R — slow, careful recovery"
        ),
        (
            "Aggressive",
            np.diag([0.025, 0.04, 11.0, 0.25]),
            np.array([[1.0]]),
            "green",
            "Low R — fast recovery"
        ),
        (
            "Hold Position",
            np.diag([1.0, 0.04, 11.0, 0.25]),
            np.array([[14.7]]),
            "orange",
            "High wheel cost"
        ),
        (
            "Smooth Motion",
            np.diag([0.025, 1.0, 11.0, 2.0]),
            np.array([[14.7]]),
            "purple",
            "High velocity cost"
        ),
    ]

    # ----------------------------------
    # Step 3: Run all experiments
    # ----------------------------------
    results = []

    print("=" * 60)
    print("Running Experiments")
    print("=" * 60)

    for name, Q, R, color, desc in experiments:

        controller = LQRController(A, B, Q, R, simulator.dt)
        K = controller.K.flatten()

        print(f"\n[{name}]")
        print(f"  Description: {desc}")
        print(f"  Q diag = {np.diag(Q)}")
        print(f"  R      = {R.flatten()}")
        print(f"  K      = {np.round(K, 4)}")

        times, states, torques = run_simulation(
            simulator, controller, INITIAL_STATE, simulator.max_ep_len
        )

        settle_time = get_settle_time(times, states)
        max_torque_used = np.max(np.abs(torques))

        print(f"  Settle time (body angle < 0.01 rad): {settle_time:.3f} s")
        print(f"  Peak torque: {max_torque_used:.4f} Nm  (limit: {simulator.max_torque:.4f} Nm)")

        results.append((name, Q, R, color, desc, times, states, torques))

    # ----------------------------------
    # Step 4: Plot — all on one figure
    # ----------------------------------
    fig, axes = make_figure(
        f"LQR Tuning Comparison  |  Initial body angle = {INITIAL_BODY_ANGLE:.2f} rad"
    )

    for name, Q, R, color, desc, times, states, torques in results:
        plot_experiment(axes, times, states, torques, name, color)

    for ax in axes:
        ax.legend(fontsize=7)

    # Mark torque limit
    axes[4].axhline( simulator.max_torque, color='k', linestyle=':', linewidth=1, label='torque limit')
    axes[4].axhline(-simulator.max_torque, color='k', linestyle=':', linewidth=1)
    axes[4].legend(fontsize=7)

    plt.tight_layout()
    plt.savefig("part1_comparison.png", dpi=150, bbox_inches='tight')
    print("\nSaved: part1_comparison.png")

    # ----------------------------------
    # Step 5: Individual plots per experiment
    # ----------------------------------
    for name, Q, R, color, desc, times, states, torques in results:

        fig2, axes2 = make_figure(f"LQR: {name}\n{desc}")

        plot_experiment(axes2, times, states, torques, name, color)

        # Add torque limit lines
        axes2[4].axhline( simulator.max_torque, color='k', linestyle=':', linewidth=1.5, label='torque limit')
        axes2[4].axhline(-simulator.max_torque, color='k', linestyle=':', linewidth=1.5)
        axes2[4].legend(fontsize=8)

        fname = f"part1_{name.replace(' ', '_').lower()}.png"
        plt.tight_layout()
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        print(f"Saved: {fname}")
        plt.close(fig2)

    plt.show()
    print("\nDone.")


# ============================================================
# Helper
# ============================================================

def get_settle_time(times, states, threshold=0.01):
    """
    Return the time at which body angle stays within threshold.
    Returns the full episode length if it never settles.
    """
    body_angles = np.abs(states[:, 2])
    for i in range(len(body_angles) - 1, -1, -1):
        if body_angles[i] > threshold:
            if i + 1 < len(times):
                return times[i + 1]
            return times[-1]
    return times[0]


if __name__ == "__main__":
    main()