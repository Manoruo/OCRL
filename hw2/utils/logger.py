import numpy as np
import matplotlib.pyplot as plt

class Logger:
    """
    Generic logger for TWIP experiments.

    Logs:
        - true state
        - estimated state (optional)
        - control input (optional)
        - time
    """

    def __init__(self):

        self.times = []
        self.true_states = []
        self.estimated_states = []
        self.controls = []

        self.has_estimate = False


    def log(self, t, true_state, estimated_state=None, control=None):

        self.times.append(t)
        self.true_states.append(true_state.copy())

        if estimated_state is not None:
            self.has_estimate = True
            self.estimated_states.append(estimated_state.copy())

        if control is not None:
            self.controls.append(control)


    def to_arrays(self):

        self.times = np.array(self.times)
        self.true_states = np.array(self.true_states)

        if self.has_estimate:
            self.estimated_states = np.array(self.estimated_states)

        if len(self.controls) > 0:
            self.controls = np.array(self.controls)


    def plot(self):

        self.to_arrays()

        labels = [
            "wheel_angle",
            "wheel_velocity",
            "body_angle",
            "body_velocity"
        ]

        fig = plt.figure(figsize=(12,8))

        gs = fig.add_gridspec(4, 2)

        # -------------------------
        # State plots
        # -------------------------

        for i in range(4):

            ax = fig.add_subplot(gs[i,0])

            ax.plot(self.times, self.true_states[:,i], label="true")

            if self.has_estimate:
                ax.plot(
                    self.times,
                    self.estimated_states[:,i],
                    "--",
                    label="estimate"
                )

            ax.set_ylabel(labels[i])
            ax.grid(True)

            if i == 0:
                ax.set_title("States")

            if i == 3:
                ax.set_xlabel("time (s)")

            if self.has_estimate:
                ax.legend()

        # -------------------------
        # Control plot
        # -------------------------

        if len(self.controls) > 0:

            ax = fig.add_subplot(gs[:,1])

            ax.plot(self.times[:len(self.controls)], self.controls)

            ax.set_title("Control (torque)")
            ax.set_xlabel("time (s)")
            ax.set_ylabel("torque")

            ax.grid(True)

        plt.tight_layout()
        plt.show(block=True)