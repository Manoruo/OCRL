# ============================================
# Learned Optimization of 4D Rosenbrock Function
# ============================================

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# --------------------------------------------
# 1. True 4D Rosenbrock (banana) function
# --------------------------------------------
def banana4(x):
    x = np.asarray(x)
    cost = 0.0
    for i in range(3):
        cost += 100.0 * (x[i + 1] - x[i]**2)**2 + (1 - x[i])**2
    return cost


# --------------------------------------------
# 2. Dataset generator with important points
# --------------------------------------------
def generate_dataset_with_important_points(
    N_uniform=8000,
    N_local=2000,
    bound=2.0,
    std=0.1,
    seed=None
):
    """
    Generates training data that includes:
    - Uniform samples in [-bound, bound]^4
    - Local Gaussian samples around known important points

    Important points:
      (1, 1, 1, 1)   global minimum
      (-1, 1, 1, 1)  local minimum
    """

    if seed is not None:
        np.random.seed(seed)

    important_points = np.array([
        [ 1.0, 1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0, 1.0]
    ])

    # Uniform samples
    X_uniform = np.random.uniform(-bound, bound, size=(N_uniform, 4))

    # Local samples around important points
    samples_per_point = N_local // len(important_points)
    X_local = []

    for p in important_points:
        Xp = np.random.normal(loc=p, scale=std, size=(samples_per_point, 4))
        X_local.append(Xp)

    X_local = np.vstack(X_local)

    # Combine datasets
    X = np.vstack([X_uniform, X_local])

    # Evaluate true function
    y = np.array([banana4(x) for x in X])

    return X, y


# --------------------------------------------
# 3. Neural network surrogate model
# --------------------------------------------
net = nn.Sequential(
    nn.Linear(4, 64),
    nn.Tanh(),
    nn.Linear(64, 64),
    nn.Tanh(),
    nn.Linear(64, 1)
)

optimizer = optim.Adam(net.parameters(), lr=3e-4)
loss_fn = nn.MSELoss()


# --------------------------------------------
# 4. Generate training data
# --------------------------------------------
X_train_np, y_train_np = generate_dataset_with_important_points(
    N_uniform=8000,
    N_local=2000,
    std=0.1,
    seed=0
)

X_train = torch.tensor(X_train_np, dtype=torch.float32)
y_train = torch.tensor(y_train_np, dtype=torch.float32).unsqueeze(-1)

# Normalize targets
y_mean = y_train.mean()
y_std = y_train.std() + 1e-8
y_train_n = (y_train - y_mean) / y_std


# --------------------------------------------
# 5. Train the surrogate
# --------------------------------------------
training_steps = 10000
for step in range(training_steps):
    optimizer.zero_grad()
    loss = loss_fn(net(X_train), y_train_n)
    loss.backward()
    optimizer.step()

    if step % 1000 == 0:
        print(f"Step {step}, loss = {loss.item():.6f}")


# --------------------------------------------
# 6. Evaluate at known important points
# --------------------------------------------
x_test = torch.tensor([
    [-1.0, 1.0, 1.0, 1.0],
    [ 1.0, 1.0, 1.0, 1.0]
], dtype=torch.float32)

y_true = torch.tensor([4.0, 0.0])

with torch.no_grad():
    y_pred_n = net(x_test).squeeze()
    y_pred = y_pred_n * y_std + y_mean

print("\nEvaluation at important points")
print("Predicted:", y_pred)
print("True:", y_true)
print("Absolute error:", torch.abs(y_pred - y_true))


# --------------------------------------------
# 7. Optimize the learned function
# --------------------------------------------
def learned_fun(x):
    with torch.no_grad():
        x_t = torch.tensor(x, dtype=torch.float32)
        y_n = net(x_t).item()
        return y_n * y_std.item() + y_mean.item()


x0 = np.random.uniform(-2, 2, size=4)

# Unconstrained optimization
res_nm = minimize(learned_fun, x0, method="Nelder-Mead")

print("\nUnconstrained optimization of learned function")
print("x* =", res_nm.x)
print("learned f(x*) =", res_nm.fun)
print("true f(x*) =", banana4(res_nm.x))


# --------------------------------------------
# 8. Constrained optimization (training domain)
# --------------------------------------------
bounds = [(-2, 2)] * 4

res_con = minimize(
    learned_fun,
    x0,
    method="L-BFGS-B",
    bounds=bounds
)

print("\nConstrained optimization of learned function")
print("x* =", res_con.x)
print("learned f(x*) =", res_con.fun)
print("true f(x*) =", banana4(res_con.x))


# --------------------------------------------
# 9. Distance to nearest training point (diagnostic)
# --------------------------------------------
X_train_t = torch.tensor(X_train_np, dtype=torch.float32)
x_star = torch.tensor([1.0, 1.0, 1.0, 1.0])

dists = torch.norm(X_train_t - x_star, dim=1)

print("\nNearest training distance to (1,1,1,1):", dists.min().item())


# --------------------------------------------
# 10. Simple 1D slice visualization
# --------------------------------------------
x1 = np.linspace(-2, 2, 400)
X_slice = np.stack([
    x1,
    np.ones_like(x1),
    np.ones_like(x1),
    np.ones_like(x1)
], axis=1)

true_vals = np.array([banana4(x) for x in X_slice])

with torch.no_grad():
    learned_vals = (
        net(torch.tensor(X_slice, dtype=torch.float32)).squeeze().numpy()
        * y_std.numpy() + y_mean.numpy()
    )

plt.figure(figsize=(6,4))
plt.plot(x1, true_vals, label="True Rosenbrock")
plt.plot(x1, learned_vals, "--", label="Learned surrogate")
plt.xlabel("$x_1$")
plt.ylabel("f(x)")
plt.title("1D slice: $x_2=x_3=x_4=1$")
plt.legend()
plt.tight_layout()
plt.show()
