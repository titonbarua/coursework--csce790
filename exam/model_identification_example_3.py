"""This module reproduces example 3 from paper -

Identification and Control of Dynamical Systems Using Neural Networks
KUMPATI S. NARENDRA FELLOW, IEEE. AND KANNAN PARTHASARATHY

Author: Titon Barua <baruat@email.sc.edu>
"""
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt


LEARNING_RATE = 0.05

RANDOM_TRAIN_STEPS = 100000
EVAL_STEPS = 300
GRAPH_Y_FILENAME = "ex3_graph_y.pdf"
GRAPH_FG_FILENAME = "ex3_graph_f_and_g.pdf"


def evolve_sys(y_hist, u_hist, calc_f, calc_g, u_k, k):
    """Evolve the dynamical system into time-step `k`.

    Args:
    - y_hist: A list of past values of plant output `y` of size `(k - 1)`.
    - u_hist: A list of past values of plant input `u` of size `(k - 1)`.
    - calc_f: A non-linear function of plant input `u`.
    - calc_g: Another non-linear function of plant input `u`.
    - u_k: Plant input at time-step `k`.
    - k: The zero-indexed time-step.

    Returns:
    A tuple of format:
       `(y_k, (f_k, f_k_inp), (g_k, g_k_inp), y_hist_updated, u_hist_updated)`
    where
      - `y_k` is the plant output.
      - `f_k` is the output of `calc_f`.
      - `f_k_inp` is the input used to generate `f_k`.
      - `g_k` is the output of `calc_g`.
      - `g_k_inp` is the input used to generate `g_k`.
      - `y_hist_updated` is the updated `y_hist`.
      - `u_hist_updated` is the updated `u_hist`.

    """
    f_k_inp = y_hist[k - 1] if k > 0 else 0.0
    f_k = calc_f(f_k_inp)

    g_k_inp = u_hist[k - 1] if k > 0 else 0.0
    g_k = calc_g(g_k_inp)

    y_k = f_k + g_k
    return (
        y_k,
        (f_k, f_k_inp),
        (g_k, g_k_inp),
        y_hist + [y_k],
        u_hist + [u_k])


def calc_f(y_k):
    """Calculate ground truth of non-linear function `f`."""
    return y_k / (1 + np.pow(y_k, 2))


def calc_g(u):
    """Calculate ground truth of non-linear function `g`."""
    return np.pow(u, 3)


# Input signals used for training and evaluation.
# ------------------------------------------------------------,
def gen_input_signal(k):
    """Generate sinusoidal input signal for the system."""
    z = 2 * np.pi * k
    return np.sin(z/25.0) + np.sin(z/10.0)
# ------------------------------------------------------------'


class Sigmoidal(nn.Module):
    """Implements a 'sigmoidal' activation function, as described in the paper."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        z = torch.exp(-x)
        return (1.0 - z) / (1.0 + z)


class LearnedFn(nn.Module):
    """Feed-forward network to approximate the non-linear function."""

    def __init__(self, device):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(1, 20),
            Sigmoidal(),
            nn.Linear(20, 10),
            Sigmoidal(),
            nn.Linear(10, 1),
            # Sigmoidal()
        )

        self.linear_stack.to(device)

    def forward(self, x):
        x = self.linear_stack(x)
        return x


def train_and_evaluate():
    net_f = LearnedFn("cpu")
    net_g = LearnedFn("cpu")
    criterion_f = nn.MSELoss()
    criterion_g = nn.MSELoss()
    optim_f = torch.optim.SGD(net_f.parameters(), lr=LEARNING_RATE)
    optim_g = torch.optim.SGD(net_g.parameters(), lr=LEARNING_RATE)

    # Generate random training data.
    # -----------------------------------------------,
    print(
        f"Generating {RANDOM_TRAIN_STEPS} "
        f"random training samples ...")
    # Choose some random input samples within (-2, 2).
    training_input = 2.0 * (np.random.rand(RANDOM_TRAIN_STEPS) - 0.5) / 0.5

    train_u_list = []
    train_y_list = []
    train_f_list = []
    train_f_inp_list = []
    train_g_list = []
    train_g_inp_list = []
    for k, u_k in enumerate(training_input):
        r = evolve_sys(
            train_y_list,
            train_u_list,
            calc_f,
            calc_g,
            u_k,
            k)

        _, (f_k, f_k_inp), (g_k, g_k_inp), train_y_list, train_u_list = r

        train_f_list.append(f_k)
        train_f_inp_list.append(f_k_inp)
        train_g_list.append(g_k)
        train_g_inp_list.append(g_k_inp)
    # -----------------------------------------------'

    # Train the networks.
    # ------------------------------------------------,
    print("Training both networks ...")

    # Convert training data to pytorch tensors and add batch dimension.
    train_f_list = torch.from_numpy(
        np.array(train_f_list, dtype=np.float32).reshape(1, -1))
    train_f_inp_list = torch.from_numpy(
        np.array(train_f_inp_list, dtype=np.float32).reshape(1, -1))
    train_g_list = torch.from_numpy(
        np.array(train_g_list, dtype=np.float32).reshape(1, -1))
    train_g_inp_list = torch.from_numpy(
        np.array(train_g_inp_list, dtype=np.float32).reshape(1, -1))

    for k in range(RANDOM_TRAIN_STEPS):
        train_f = train_f_list[:, k]
        train_f_inp = train_f_inp_list[:, k]
        train_g = train_g_list[:, k]
        train_g_inp = train_g_inp_list[:, k]

        optim_f.zero_grad()
        optim_g.zero_grad()

        pred_f = net_f(train_f_inp)
        pred_g = net_g(train_g_inp)

        loss_f = criterion_f(pred_f, train_f)
        loss_g = criterion_g(pred_g, train_g)

        loss_f.backward()
        loss_g.backward()

        optim_f.step()
        optim_g.step()
    # ------------------------------------------------'

    # Test identified model.
    # ------------------------------------------------,
    print("Evaluating identified model ...")
    time_steps = range(EVAL_STEPS)

    # Generate test input and output signals.
    eval_u_list = []
    eval_y_list = []
    for k in time_steps:
        u_k = gen_input_signal(k)
        _, _, _, eval_y_list, eval_u_list = evolve_sys(
            eval_y_list,
            eval_u_list,
            calc_f,
            calc_g,
            u_k,
            k)

    # Predict output signal using the neural networks.
    pred_u_list = []
    pred_y_list = []
    for k in time_steps:
        u_k = eval_u_list[k]
        _, _, _, pred_y_list, pred_u_list = evolve_sys(
            pred_y_list,
            pred_u_list,
            lambda x: net_f(torch.tensor(x, dtype=torch.float32).reshape(1, 1)).item(),
            lambda x: net_g(torch.tensor(x, dtype=torch.float32).reshape(1, 1)).item(),
            u_k,
            k)

    # Plot graph of true y vs predicted y.
    ax = plt.subplot()
    ax.set_title(
        "Example 3: Output ground truth $y$ vs prediction $\^{y}$",
        fontsize=10)
    ax.plot(time_steps, eval_y_list, label="y")
    ax.plot(time_steps, pred_y_list, label="$\^{y}$", linestyle="dashed")
    ax.set_xlabel("Timestep, $k$")
    ax.set_ylabel("y vs $\^{y}$")
    ax.figure.set_size_inches(8, 3)
    ax.legend()
    ax.figure.tight_layout()
    ax.figure.savefig(GRAPH_Y_FILENAME)
    plt.show()

    # Plot graph of true non-linear functions vs their NN approximation.
    # ------------------------------------------------------------------,
    f_x = np.linspace(-10, 10, 100)
    f_real = [calc_f(x) for x in f_x]
    f_approx = [
        net_f(torch.tensor(x, dtype=torch.float32).reshape(1, 1)).item()
        for x in f_x]

    g_x = np.linspace(-2, 2, 100)
    g_real = [calc_g(x) for x in g_x]
    g_approx = [
        net_g(torch.tensor(x, dtype=torch.float32).reshape(1, 1)).item()
        for x in g_x]

    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    ax_f, ax_g = axes[0], axes[1]
    ax_f.set_title(
        "Example 3: $f(y) = y / (1 + y^2)$ vs $N_f$",
        fontsize=10)
    ax_f.plot(f_x, f_real, label="$f(y)$")
    ax_f.plot(f_x, f_approx, label="$N_f$", linestyle="dashed")
    ax_f.set_xlabel("$y$")
    ax_f.set_ylabel("$f(y)$ vs $N_f$")
    ax_f.legend()

    ax_g.set_title(
        "Example 3: $g(u) = u^3$ vs $N_g$",
        fontsize=10)
    ax_g.plot(g_x, g_real, label="$g(u)$")
    ax_g.plot(g_x, g_approx, label="$N_g$", linestyle="dashed")
    ax_g.set_xlabel("$u$")
    ax_g.set_ylabel("$g(u)$ vs $N_g$")
    ax_g.legend()

    fig.tight_layout()
    fig.savefig(GRAPH_FG_FILENAME)
    plt.show()
    # ------------------------------------------------------------------'


if __name__ == "__main__":
    train_and_evaluate()
