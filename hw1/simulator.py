import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt



# ============================================================
# Two-Wheeled Inverted Pendulum Environment
# ============================================================

class TWIPEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, params):
        super().__init__()

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
        self.time_step = params["dt"]
        self.max_body_angle = 1.5
        self.max_torque = params["max_torque"]

        self.initial_state_range = params["x_initial_range"]

        # --------------------
        # Gym spaces
        # --------------------
        self.action_space = spaces.Box(
            low=-self.max_torque,
            high=self.max_torque,
            shape=(1,),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, -np.pi, -np.inf], dtype=np.float32),
            high=np.array([np.inf, np.inf, np.pi, np.inf], dtype=np.float32),
        )

        self.state = None

    # --------------------
    # Reset (Gym ≥0.26)
    # --------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        r = self.initial_state_range
        self.state = np.array([
            np.random.uniform(*r["wh"]),
            np.random.uniform(*r["whd"]),
            np.random.uniform(*r["th"]),
            np.random.uniform(*r["thd"]),
        ], dtype=np.float32)

        return self.state.copy(), {}

    # --------------------
    # Dynamics
    # --------------------
    def twip3(
        self,
        wheel_angle,
        wheel_velocity,
        body_angle,
        body_angular_velocity,
        applied_torque,
    ):
        p1 = self.wheel_inertia + (self.body_mass + self.wheel_mass) * self.wheel_radius**2
        p2 = self.body_mass * self.body_com_length
        p3 = self.body_inertia + self.body_mass * self.body_com_length**2

        coupling = p1 + p2 * self.wheel_radius * np.cos(body_angle)

        M = np.array([
            [p1, coupling],
            [coupling, p1 + 2 * p2 * self.wheel_radius * np.cos(body_angle) + p3],
        ])

        coriolis = p2 * self.wheel_radius * body_angular_velocity**2 * np.sin(body_angle)

        rhs = np.array([
            -applied_torque - self.motor_friction * wheel_velocity + coriolis,
            coriolis + p2 * self.gravity * np.sin(body_angle),
        ])

        wheel_accel, body_accel = np.linalg.solve(M, rhs)
        return wheel_accel, body_accel

    # --------------------
    # Step (Gym ≥0.26)
    # --------------------
    def step(self, action):
        torque = float(np.clip(action[0], -self.max_torque, self.max_torque))

        wheel_angle = self.state[0]
        wheel_velocity = self.state[1]
        body_angle = self.state[2]
        body_angular_velocity = self.state[3]

        wheel_accel, body_accel = self.twip3(
            wheel_angle,
            wheel_velocity,
            body_angle,
            body_angular_velocity,
            torque,
        )

        wheel_velocity += wheel_accel * self.time_step
        wheel_angle += wheel_velocity * self.time_step

        body_angular_velocity += body_accel * self.time_step
        body_angle += body_angular_velocity * self.time_step

        self.state = np.array(
            [wheel_angle, wheel_velocity, body_angle, body_angular_velocity],
            dtype=np.float32,
        )

        terminated = abs(body_angle) > self.max_body_angle
        truncated = False

        reward = -(body_angle**2 + 0.1 * body_angular_velocity**2)
        if terminated:
            reward -= 10.0

        return self.state.copy(), reward, terminated, truncated, {}

    # --------------------
    # Rendering
    # --------------------
    def render(self):
        wheel_angle, wheel_velocity, body_angle, body_angular_velocity = self.state

        axle_x = 0.0
        axle_y = self.wheel_radius
        L = self.body_com_length

        tip_x = axle_x + L * np.sin(body_angle)
        tip_y = axle_y + L * np.cos(body_angle)

        if not hasattr(self, "_fig"):
            plt.ion()
            self._fig, self._ax = plt.subplots()
            self._ax.set_xlim(-0.3, 0.3)
            self._ax.set_ylim(0.0, 0.4)
            self._ax.set_aspect("equal")

            self._ax.plot([-1, 1], [0, 0], "k", linewidth=2)

            self._wheel = plt.Circle((axle_x, axle_y), self.wheel_radius, fill=False)
            self._ax.add_patch(self._wheel)

            self._ax.plot(axle_x, axle_y, "ko", markersize=6)

            self._upright, = self._ax.plot(
                [axle_x, axle_x],
                [axle_y, axle_y + L],
                "k--",
                alpha=0.4,
            )

            self._body, = self._ax.plot(
                [axle_x, tip_x],
                [axle_y, tip_y],
                "r-",
                linewidth=3,
            )

        self._body.set_data([axle_x, tip_x], [axle_y, tip_y])
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

# ============================================================
# Linear feedback controller (policy)
# ============================================================

def linear_feedback_controller(state, gains):
    """
    Decide how much torque to apply based on how the robot is moving.

    Control law. Negative sign telling you to push in opposite direction in which ur falling.

    state  = [wheel_angle, wheel_velocity, body_angle, body_angular_velocity]
    gains  = [k_wheel_angle, k_wheel_velocity, k_body_angle, k_body_velocity]

    Each gain tells us how strongly to react to that part of the state.
    """

    wheel_angle = state[0]
    wheel_velocity = state[1]
    body_angle = state[2]
    body_angular_velocity = state[3]

    k_wheel_angle = gains[0]
    k_wheel_velocity = gains[1]
    k_body_angle = gains[2]
    k_body_angular_velocity = gains[3]

    # Compute how much torque to apply:
    # - push back against body lean
    # - damp falling motion
    # - lightly damp wheel motion
    torque = (
        -k_wheel_angle * wheel_angle # p term 
        -k_wheel_velocity * wheel_velocity # D term
        -k_body_angle * body_angle # p term
        -k_body_angular_velocity * body_angular_velocity # D term
    )

    return torque
