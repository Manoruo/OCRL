import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt
import yaml
from pathlib import Path


# ============================================================
# Two-Wheeled Inverted Pendulum Environment
# ============================================================

class TWIPEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, sim_params_path=None):
        super().__init__()

        params = self._load_params(sim_params_path)

        # --------------------
        # Physical parameters
        # --------------------
        self.wheel_mass = params["m_w"]
        self.wheel_radius = params["r_w"]
        self.wheel_inertia = params["I_w"]

        self.body_mass = params["m_p"]
        self.body_com_length = params["l_p"]
        self.body_inertia = params["I_p"]

        self.gravity = params["g"]
        self.motor_friction = params["f"]

        # --------------------
        # Simulation parameters
        # --------------------
        self.dt = params["dt"]
        self.max_torque = params["max_torque"]
        self.max_body_angle = params["max_body_angle"]
        self.max_ep_len = params["ep_len"] 

        self.initial_state_range = params["x_initial_range"]

        # --------------------
        # Gym spaces
        # --------------------
        # Action = feedback gains
        # [k_wheel_angle, k_wheel_velocity, k_body_angle, k_body_angular_velocity]
        self.action_space = spaces.Box(
            low=np.array([-50.0, -50.0, -200.0, -50.0], dtype=np.float32),
            high=np.array([50.0, 50.0, 200.0, 50.0], dtype=np.float32),
        )

        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, -np.pi, -np.inf], dtype=np.float32),
            high=np.array([np.inf, np.inf, np.pi, np.inf], dtype=np.float32),
        )

        self.state = None
        self.logs = {}
        self.step_count = 0
        self.time = 0.0

    # --------------------------------------------------------
    # Parameter loading
    # --------------------------------------------------------

    def _load_params(self, path):
        if path is None:
            print("[TWIPEnv] No sim_params_path provided → using default parameters")
            return self._default_params()

        path = Path(path)
        if not path.exists():
            print(f"[TWIPEnv] sim_params file not found: {path} → using defaults")
            return self._default_params()

        try:
            with open(path, "r") as f:
                params = yaml.safe_load(f)
            if params is None:
                raise ValueError("empty YAML")
            print(f"[TWIPEnv] Loaded sim params from: {path}")
            return params
        except Exception as e:
            print(f"[TWIPEnv] Failed to load sim params ({e}) → using defaults")
            return self._default_params()

    def _default_params(self):
        return {
            "m_w": 0.173,
            "r_w": 0.0323,
            "I_w": 0.0066,
            "m_p": 0.653,
            "l_p": 0.043,
            "I_p": 0.00084,
            "g": 9.81,
            "f": 0.0,
            "dt": 1 / 333,
            "max_torque": 5.0,
            "max_body_angle": 1.5,
            "ep_len": 3,
            "x_initial_range": {
                "wh": [-6.28, 6.28],
                "whd": [-8.0, 8.0],
                "th": [-0.35, 0.35],
                "thd": [-6.0, 6.0],
            },
        }

    # --------------------------------------------------------
    # Reset
    # --------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Default range from config
        r = self.initial_state_range

        # Allow override via options
        if options is not None and "x_initial_range" in options:
            r = options["x_initial_range"]

        self.state = np.array(
            [
                np.random.uniform(*r["wh"]),   # wheel angle [rad]
                np.random.uniform(*r["whd"]),  # wheel angular velocity [rad/s]
                np.random.uniform(*r["th"]),   # body angle [rad]
                np.random.uniform(*r["thd"]),  # body angular velocity [rad/s]
            ],
            dtype=np.float32,
        )

        self.logs = {
            "time": [],
            "wheel_angle": [],
            "wheel_velocity": [],
            "body_angle": [],
            "body_angular_velocity": [],
            "torque": [],
            "gains": [],
        }

        self.step_count = 0
        self.time = 0.0

        return self.state.copy(), {}


    # --------------------------------------------------------
    # Dynamics
    # --------------------------------------------------------

    def twip3(self, wh, whd, th, thd, trq):
        """
        Forward dynamics of TWIP (MATLAB-matching notation)

        State:
            wh   : wheel angle (rad)              [not used in dynamics]
            whd  : wheel angular velocity (rad/s)
            th   : body angle from vertical (rad)
            thd  : body angular velocity (rad/s)

        Input:
            trq  : applied wheel torque (Nm)

        Returns:
            whdd : wheel angular acceleration (rad/s^2)
            thdd : body angular acceleration (rad/s^2)
        """

        # --- Composite inertial terms ---
        p1 = self.wheel_inertia + (self.body_mass + self.wheel_mass) * self.wheel_radius**2
        p2 = self.body_mass * self.body_com_length
        p3 = self.body_inertia + self.body_mass * self.body_com_length**2

        # --- Mass matrix ---
        m12 = p1 + p2 * self.wheel_radius * np.cos(th)
        M = np.array([
            [p1, m12],
            [m12, p1 + 2.0 * p2 * self.wheel_radius * np.cos(th) + p3],
        ])

        # --- Velocity-dependent term ---
        v = p2 * self.wheel_radius * thd**2 * np.sin(th)

        # --- RHS ---
        rhs = np.array([
            -trq - self.motor_friction * whd + v,
            v + p2 * self.gravity * np.sin(th),
        ])

        # --- Solve for accelerations ---
        whdd, thdd = np.linalg.solve(M, rhs)

        return whdd, thdd


    # --------------------------------------------------------
    # Controller (explicit + readable)
    # --------------------------------------------------------

    def feedback_control(self, state, gains):
        """
        gains =
            [k_wheel_angle,
             k_wheel_velocity,
             k_body_angle,
             k_body_angular_velocity]
        """

        wheel_angle = state[0]
        wheel_velocity = state[1]
        body_angle = state[2]
        body_angular_velocity = state[3]

        k_wheel_angle = gains[0]
        k_wheel_velocity = gains[1]
        k_body_angle = gains[2]
        k_body_angular_velocity = gains[3]

        torque = (
            -k_wheel_angle * wheel_angle
            -k_wheel_velocity * wheel_velocity
            -k_body_angle * body_angle
            -k_body_angular_velocity * body_angular_velocity
        )

        return torque

    # --------------------------------------------------------
    # Step
    # --------------------------------------------------------

    def step(self, action, log=False):
        gains = np.asarray(action, dtype=np.float32)

        torque = self.feedback_control(self.state, gains)
        torque = float(np.clip(torque, -self.max_torque, self.max_torque))

        (
            wheel_angle,
            wheel_velocity,
            body_angle,
            body_angular_velocity,
        ) = self.state

        wheel_accel, body_accel = self.twip3(
            wheel_angle,
            wheel_velocity,
            body_angle,
            body_angular_velocity,
            torque,
        )

        # Euler integration
        wheel_velocity += wheel_accel * self.dt
        wheel_angle += wheel_velocity * self.dt

        body_angular_velocity += body_accel * self.dt
        body_angle += body_angular_velocity * self.dt

        self.state = np.array(
            [
                wheel_angle,
                wheel_velocity,
                body_angle,
                body_angular_velocity,
            ],
            dtype=np.float32,
        )

        self.step_count += 1
        self.time += self.dt

        terminated = abs(body_angle) > self.max_body_angle
        truncated = self.time >= self.max_ep_len  # CHANGED: use max_ep_len

        reward = -(body_angle**2 + 0.1 * body_angular_velocity**2)
        if terminated:
            reward -= 10.0

        if log:
            self.logs["time"].append(self.time)
            self.logs["wheel_angle"].append(wheel_angle)
            self.logs["wheel_velocity"].append(wheel_velocity)
            self.logs["body_angle"].append(body_angle)
            self.logs["body_angular_velocity"].append(body_angular_velocity)
            self.logs["torque"].append(torque)
            self.logs["gains"].append(gains.copy())

        return self.state.copy(), reward, terminated, truncated, {}

    # --------------------------------------------------------
    # Plot logs
    # --------------------------------------------------------

    def plot_logs(self):
        t = np.array(self.logs["time"])

        wh  = np.array(self.logs["wheel_angle"])
        whd = np.array(self.logs["wheel_velocity"])
        th  = np.array(self.logs["body_angle"])
        thd = np.array(self.logs["body_angular_velocity"])

        fig, axs = plt.subplots(2, 2, figsize=(10, 6), sharex=True)

        axs[0, 0].plot(t, wh)
        axs[0, 0].set_ylabel("Wheel angle [rad]")
        axs[0, 0].grid(True)

        axs[1, 0].plot(t, whd)
        axs[1, 0].set_ylabel("Wheel vel [rad/s]")
        axs[1, 0].set_xlabel("Time [s]")
        axs[1, 0].grid(True)

        axs[0, 1].plot(t, th)
        axs[0, 1].set_ylabel("Body angle [rad]")
        axs[0, 1].grid(True)

        axs[1, 1].plot(t, thd)
        axs[1, 1].set_ylabel("Body vel [rad/s]")
        axs[1, 1].set_xlabel("Time [s]")
        axs[1, 1].grid(True)

        plt.tight_layout()
        plt.show(block=True)


    # --------------------------------------------------------
    # Render
    # --------------------------------------------------------

    def render(self):
        (
            wheel_angle,
            wheel_velocity,
            body_angle,
            body_angular_velocity,
        ) = self.state

        axle_x = 0.0
        axle_y = self.wheel_radius
        body_length = self.body_com_length

        tip_x = axle_x + body_length * np.sin(body_angle)
        tip_y = axle_y + body_length * np.cos(body_angle)

        if not hasattr(self, "_fig"):
            plt.ion()
            self._fig, self._ax = plt.subplots()

            # ADDED: title
            self._ax.set_title("Two-Wheeled Inverted Pendulum")

            self._ax.set_xlim(-0.3, 0.3)
            self._ax.set_ylim(0.0, 0.4)
            self._ax.set_aspect("equal")

            # Ground
            self._ax.plot([-1, 1], [0, 0], "k", linewidth=2)

            # Wheel
            self._wheel = plt.Circle(
                (axle_x, axle_y),
                self.wheel_radius,
                fill=False,
            )
            self._ax.add_patch(self._wheel)

            # Axle
            self._ax.plot(axle_x, axle_y, "ko", markersize=6)

            # Upright goal (dotted)
            self._upright, = self._ax.plot(
                [axle_x, axle_x],
                [axle_y, axle_y + body_length],
                "k--",
                alpha=0.4,
            )

            # Body
            self._body, = self._ax.plot(
                [axle_x, tip_x],
                [axle_y, tip_y],
                "r-",
                linewidth=3,
            )

            # ADDED: timer text (elapsed / total)
            self._time_text = self._ax.text(
                0.02, 0.95, "",
                transform=self._ax.transAxes,
                fontsize=10,
                verticalalignment="top",
            )

        self._body.set_data([axle_x, tip_x], [axle_y, tip_y])

        # ADDED: update timer text
        self._time_text.set_text(
            f"t = {self.time:.2f} / {self.max_ep_len:.2f} s"
        )

        self._fig.canvas.draw()
        self._fig.canvas.flush_events()
