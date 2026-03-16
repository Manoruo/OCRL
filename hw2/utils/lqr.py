import numpy as np
from scipy.linalg import solve_discrete_are


class LQRController:

    def __init__(self, A, B, Q, R, dt, x_eq=None):

        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.dt = dt

        if x_eq is None:
            self.x_eq = np.zeros(A.shape[0])
        else:
            self.x_eq = x_eq

        # Convert to discrete system
        self.Ad = np.eye(A.shape[0]) + A * dt
        self.Bd = B * dt

        # Solve Riccati equation
        self.P = solve_discrete_are(self.Ad, self.Bd, Q, R)

        # Compute optimal gain
        self.K = np.linalg.inv(self.Bd.T @ self.P @ self.Bd + R) @ (self.Bd.T @ self.P @ self.Ad)

    def control(self, x):
        x_err = x - self.x_eq
        u = -self.K @ x_err
        return float(u)