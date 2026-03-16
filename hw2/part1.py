from utils.simulator import TWIPEnv
from utils.logger import Logger
import numpy as np
from utils.lqr import LQRController

SIM_PARAMS_PATH = "./sim_params.yaml"
simulator = TWIPEnv(sim_params_path=SIM_PARAMS_PATH, control_mode="torque")

logger = Logger()

# ---------------------------------
# Linearize system
# ---------------------------------
A, B = simulator.linearize()
print("A =\n", A)
print("\nB =\n", B)

# ---------------------------------
# LQR cost matrices
# ---------------------------------
Q = np.diag([
    1.0,    # wheel angle
    1.0,    # wheel velocity
    100.0,  # body angle
    10.0    # body angular velocity
])
R = np.array([[0.1]])

# ---------------------------------
# Create controller
# ---------------------------------
controller = LQRController(A, B, Q, R, simulator.dt)
print("K =", controller.K)
gains = controller.K.flatten()

# ---------------------------------
# Run simulation
# ---------------------------------
state, _ = simulator.reset()

render_every = 10

for i in range(2000):
    # Log state and gains
    torque = controller.control(state)
    logger.log(simulator.time, state, control=torque)
    
    state, reward, _, trunc, _ = simulator.step(torque, )
    
    if i % render_every == 0:
        simulator.render()
    
    if trunc:
        break

logger.plot()
