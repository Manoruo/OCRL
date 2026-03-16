import numpy as np

from utils.logger import Logger
from utils.sense import Sensor, KalmanFilter
from utils.simulator import TWIPEnv
from utils.lqr import LQRController


SIM_PARAMS_PATH = "./sim_params.yaml"

RENDER = False
render_every = 2


# ---------------------------------
# Create simulator
# ---------------------------------

simulator = TWIPEnv(sim_params_path=SIM_PARAMS_PATH, control_mode="torque")


# ---------------------------------
# Linearize system
# ---------------------------------

A, B = simulator.linearize()


# ---------------------------------
# LQR controller
# ---------------------------------

Q = np.diag([1.0, 1.0, 100.0, 10.0])
R = np.array([[0.1]])

controller = LQRController(A, B, Q, R, simulator.dt)


# ---------------------------------
# Discrete dynamics
# ---------------------------------

dt = simulator.dt
Ad = np.eye(4) + A * dt
Bd = B * dt


# ---------------------------------
# Measurement matrix
# ---------------------------------

C = np.array([
    [1,0,0,0],
    [0,0,1,0],
    [0,0,0,1]
])


# ---------------------------------
# Process noise
# ---------------------------------

Qf = np.diag([
    1e-4,
    1e-3,
    1e-4,
    1e-3
])


# ---------------------------------
# Experiments
# ---------------------------------

experiments = {
    "baseline": np.diag([1e-2, 1e-2, 1e-2]),
    "high_measurement_noise": np.diag([1e-1, 1e-1, 1e-1]),
    "low_measurement_noise": np.diag([1e-4, 1e-4, 1e-4]),
    "noisy_accelerometer": np.diag([1e-2, 1e-1, 1e-2])
}


for name, Rf in experiments.items():

    print("\nRunning experiment:", name)

    logger = Logger()
    state, _ = simulator.reset()
    kf = KalmanFilter(Ad, Bd, C, Qf, Rf)
    #kf.x_hat = state.copy() # Assume first step we have the true dynamics
    

    u = 0

    for i in range(2000):

        t = simulator.time

        # measurement
        y = Sensor.get_measurement(state, simulator.gravity)

        # kalman update
        x_hat = kf.step(u, y)

        # control
        u = controller.control(x_hat)

        # simulate
        state, reward, _, trunc, _ = simulator.step(u)

        logger.log(
            t=t,
            true_state=state,
            estimated_state=x_hat,
            control=u
        )

        if RENDER and i % render_every == 0:
            simulator.render()

        if trunc:
            break


    print("Plotting:", name)

    logger.plot()