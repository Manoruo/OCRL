import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from simulator import TWIPEnv


# ============================================================
# Constants
# ============================================================

SIM_PARAMS_PATH = "./sim_params.yaml"
MODEL_PATH = "nn_twip_gain_policy.pt"

KC_STAR = np.array([0.0, 0.5, 10.0, 1.05], dtype=np.float32)

N_SAMPLES  = 5_000
BATCH_SIZE = 256
EPOCHS     = 5
LR         = 1e-3

VISUAL_RESET_OPTIONS = {
    "x_initial_range": {
        "wh":  [np.deg2rad(180), np.deg2rad(180)],
        "whd": [0.0, 0.0],
        "th":  [np.deg2rad(45), np.deg2rad(45)],
        "thd": [np.deg2rad(140), np.deg2rad(140)],
    }
}


# ============================================================
# Neural Network Gain Policy
# ============================================================

class GainPolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    env = TWIPEnv(SIM_PARAMS_PATH)

    policy = GainPolicyNet()

    # --------------------------------------------------------
    # Load or train
    # --------------------------------------------------------
    if os.path.exists(MODEL_PATH):
        policy.load_state_dict(torch.load(MODEL_PATH))
        policy.eval()
        print("Loaded existing gain policy.")
    else:
        # Dataset
        X, Y = [], []

        for _ in range(N_SAMPLES):
            state, _ = env.reset()
            X.append(state)
            Y.append(KC_STAR)

        X = torch.as_tensor(np.array(X), dtype=torch.float32)
        Y = torch.as_tensor(np.array(Y), dtype=torch.float32)

        optimizer = optim.Adam(policy.parameters(), lr=LR)
        loss_fn = nn.MSELoss()

        for epoch in range(EPOCHS):
            perm = torch.randperm(len(X))
            epoch_loss = 0.0

            for i in range(0, len(X), BATCH_SIZE):
                idx = perm[i:i + BATCH_SIZE]
                pred = policy(X[idx])
                loss = loss_fn(pred, Y[idx])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            print(f"Epoch {epoch:02d} | Loss = {epoch_loss:.3e}")

        torch.save(policy.state_dict(), MODEL_PATH)
        print("Saved gain policy.")

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------
    state, _ = env.reset(options=VISUAL_RESET_OPTIONS)

    while True:
        with torch.no_grad():
            gains = policy(
                torch.as_tensor(state, dtype=torch.float32)
            ).numpy()

        state, _, terminated, truncated, _ = env.step(gains, log=True)
        env.render()

        if terminated or truncated:
            env.plot_logs()
            break
