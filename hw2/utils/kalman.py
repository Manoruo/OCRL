
import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter:
    """
    Kalman Filter for estimating the TWIP state.

    Estimated state:
        x_hat =
        [
            wheel_angle,
            wheel_angular_velocity,
            body_angle,
            body_angular_velocity
        ]

    Measurement:
        y =
        [
            wheel_encoder,
            accelerometer_signal,
            gyro
        ]
    """

    def __init__(self, Ad, Bd, C, Qf, Rf):

        # discrete system dynamics
        self.Ad = np.asarray(Ad, dtype=float)
        self.Bd = np.asarray(Bd, dtype=float)

        # measurement model
        self.C = np.asarray(C, dtype=float)

        # filter tuning
        self.Qf = np.asarray(Qf, dtype=float)
        self.Rf = np.asarray(Rf, dtype=float)

        n = self.Ad.shape[0]

        # current state estimate
        self.x_hat = np.zeros(n, dtype=float)

        # estimate covariance
        self.P = np.eye(n, dtype=float)

    def predict(self, u):
        """
        Prediction step:
            x_hat <- Ad x_hat + Bd u
            P     <- Ad P Ad^T + Qf
        """

        u_vec = np.array([u], dtype=float)
        self.x_hat = self.Ad @ self.x_hat + (self.Bd @ u_vec).flatten()

        self.P = self.Ad @ self.P @ self.Ad.T + self.Qf

    def update(self, y):
        """
        Measurement update step.
        """

        y = np.asarray(y, dtype=float)

        # predicted measurement
        y_pred = self.C @ self.x_hat

        # innovation = actual measurement - predicted measurement
        innovation = y - y_pred

        # innovation covariance
        S = self.C @ self.P @ self.C.T + self.Rf

        # Kalman gain
        K = self.P @ self.C.T @ np.linalg.inv(S)

        # update estimate
        self.x_hat = self.x_hat + K @ innovation

        # update covariance
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.C) @ self.P

    def step(self, u, y):
        """
        One full Kalman filter step.
        """

        self.predict(u)
        self.update(y)

        return self.x_hat.copy()

