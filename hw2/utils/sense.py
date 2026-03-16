
import numpy as np
import matplotlib.pyplot as plt

class Sensor:

    @staticmethod
    def get_measurement(state, gravity, noise_std=0.01):
        """
        Returns measurement vector y for the homework.

        State =
        [
            wheel_angle,
            wheel_angular_velocity,
            body_angle,
            body_angular_velocity
        ]

        Measurement =
        [
            wheel_encoder,
            accelerometer_signal,
            gyro
        ]

        We model accelerometer_signal as a body-angle estimate:
            ax = g * sin(body_angle)
            az = g * cos(body_angle)
            body_angle_from_accelerometer = atan2(ax, az)
        """

        wheel_angle = state[0]
        body_angle = state[2]
        body_angular_velocity = state[3]

        # accelerometer signals in body frame
        ax = gravity * np.sin(body_angle)
        az = gravity * np.cos(body_angle)

        # accelerometer-derived body angle measurement
        body_angle_from_accelerometer = np.arctan2(ax, az)

        measurement = np.array([
            wheel_angle,
            body_angle_from_accelerometer,
            body_angular_velocity
        ], dtype=float)

        noise = np.random.normal(0, noise_std, size=3)

        return measurement + noise

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

