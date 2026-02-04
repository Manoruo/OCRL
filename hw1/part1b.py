import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.optimize import minimize

torch.manual_seed(0)
np.random.seed(0)


# ============================================================
# True 4D Rosenbrock
# ============================================================
def banana4(x):
    x = np.asarray(x)
    return np.sum(
        100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2
    )


# ============================================================
# Dataset generation
# ============================================================
def generate_training_data(N, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2.0, 2.0, size=(N, 4))
    y = np.array([banana4(x) for x in X])
    return X, y


# ============================================================
# Neural network
# ============================================================
def make_net():
    return nn.Sequential(
        nn.Linear(4, 32),
        nn.Tanh(),
        nn.Linear(32, 32),
        nn.Tanh(),
        nn.Linear(32, 1),
    )


def train_net(net, X, y, steps=100000, lr=1/100):
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

    opt = optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn(net(X_t), y_t)
        loss.backward()
        opt.step()


# ============================================================
# Learned function wrapper
# ============================================================
def learned_fun(net):
    def f(x):
        with torch.no_grad():
            x_t = torch.tensor(x, dtype=torch.float32)
            return net(x_t).item()
    return f


# ============================================================
# Experiment for different dataset sizes
# ============================================================
dataset_sizes = [50, 100, 1000, 10000]
results = []

os.makedirs("models", exist_ok=True)

for N in dataset_sizes:
    print(f"\n=== N = {N} ===")
    model_path = f"models/banana_net_N{N}.pt"

    X, y = generate_training_data(N, seed=0)
    net = make_net()

    if os.path.exists(model_path):
        net.load_state_dict(torch.load(model_path))
        net.eval()
        print("Loaded model.")
    else:
        train_net(net, X, y)
        torch.save(net.state_dict(), model_path)
        print("Trained and saved model.")

    # Evaluate at key points
    x_test = np.array([[1, 1, 1, 1], [-1, 1, 1, 1]])
    y_true = np.array([banana4(x) for x in x_test])

    with torch.no_grad():
        y_pred = net(torch.tensor(x_test, dtype=torch.float32)).squeeze().numpy()

    print("Predicted:", y_pred)
    print("True:     ", y_true)

    # Optimize learned function
    f_hat = learned_fun(net)
    x0 = np.random.uniform(-2, 2, size=4)

    res_uncon = minimize(f_hat, x0, method="Nelder-Mead")
    res_con = minimize(
        f_hat,
        x0,
        method="L-BFGS-B",
        bounds=[(-2, 2)] * 4
    )

    results.append((N, net, res_uncon, res_con))


# ============================================================
# Visualization 1: Value fit scatter (largest dataset)
# ============================================================
N, net, _, _ = results[-1]
X, y = generate_training_data(N)

with torch.no_grad():
    y_hat = net(torch.tensor(X, dtype=torch.float32)).squeeze().numpy()

plt.figure(figsize=(4, 4))
plt.scatter(y, y_hat, s=3, alpha=0.3)
plt.plot([0, y.max()], [0, y.max()], 'r--')
plt.xlabel("True f(x)")
plt.ylabel("Learned f̂(x)")
plt.title("Value Fit (Uniform Data)")
plt.tight_layout()
plt.show()


# ============================================================
# Visualization 2: 1D slice through valley
# ============================================================
x1 = np.linspace(-2, 2, 400)
X_slice = np.stack(
    [x1, np.ones_like(x1), np.ones_like(x1), np.ones_like(x1)],
    axis=1,
)

true_vals = np.array([banana4(x) for x in X_slice])
with torch.no_grad():
    learned_vals = net(torch.tensor(X_slice, dtype=torch.float32)).squeeze().numpy()

plt.figure(figsize=(6, 4))
plt.plot(x1, true_vals, label="True Rosenbrock")
plt.plot(x1, learned_vals, "--", label="Learned")
plt.xlabel("$x_1$")
plt.ylabel("f(x)")
plt.title("1D Slice: $x_2=x_3=x_4=1$")
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# Visualization 3: Optimization behavior
# ============================================================
for N, net, res_uncon, res_con in results:
    print(f"\n=== Optimization results (N={N}) ===")
    print("Unconstrained x*:", res_uncon.x, "f:", banana4(res_uncon.x))
    print("Constrained   x*:", res_con.x,   "f:", banana4(res_con.x))
