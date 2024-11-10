"""This module reproduces example 1 from paper -

Identification and Control of Dynamical Systems Using Neural Networks
KUMPATI S. NARENDRA FELLOW, IEEE. AND KANNAN PARTHASARATHY

Author: Titon Barua <baruat@email.sc.edu>
"""

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt


LEARNING_RATE = 0.25

FN1_SIGA_SYNC_TRAIN_STEPS = 500
FN1_SIGA_EVAL_STEPS = 700
FN1_GRAPH_FILENAME = "ex1_fn1.pdf"

FN2_SIGA_RANDOM_TRAIN_STEPS = 50000
FN2_SIGA_EVAL_STEPS = 250
FN2_SIGB_EVAL_STEPS = 250
FN2_GRAPH1_FILENAME = "ex1_fn2.pdf"
FN2_GRAPH2_FILENAME = "ex1_fn2_modified_signal.pdf"


def evolve_sys(y_hist, u_hist, calc_f, u_k, k):
    """Evolve the dynamical system into next time step.

    Args:
    - y_hist: A list of past values of plant output `y` of size `(k - 1)`.
    - u_hist: A list of past values of plant input `u` of size `(k - 1)`.
    - calc_f: A Non-linear function of plant input `u`.
    - u_k: Plant input at current time step.
    - k: The zero-indexed time-step.

    Returns:
    A tuple of format: `(y_k, (f_k, f_k_inp), y_hist_updated, u_hist_updated)`,
    where
      - `y_k` is the plant output.
      - `f_k` is the output of `calc_f`.
      - `f_k_inp` is the input used to generate `f_k`.
      - `y_hist_updated` is the updated `y_hist`.
      - `u_hist_updated` is the updated `u_hist`.

    """
    past_y1 = y_hist[k - 1] if k > 0 else 0.0
    past_y2 = y_hist[k - 2] if k > 1 else 0.0

    f_k_inp = u_hist[k - 1] if k > 0 else 0.0
    f_k = calc_f(f_k_inp)

    y_k = 0.3 * past_y1 + 0.6 * past_y2 + f_k

    return (y_k, (f_k, f_k_inp), y_hist + [y_k], u_hist + [u_k])


def calc_non_linear_fn1(u):
    """Calculate ground truth of non-linear function."""
    pi_u = np.pi * u
    return (
        0.6 * np.sin(pi_u) +
        0.3 * np.sin(3 * pi_u) +
        0.1 * np.sin(5 * pi_u))


def calc_non_linear_fn2(u):
    """Calculate ground truth of non-linear function."""
    return (u * u * u) + 0.3 * (u * u) - 0.4 * u


# Input signals used for training and evaluation.
# ------------------------------------------------------------,
def gen_input_signal_a(k):
    """Generate sinusoidal input signal for the system."""
    return np.sin((2.0 * np.pi * float(k)) / 250.0)


def gen_input_signal_b(k):
    """Generate sinusoidal input signal for the system."""
    z = 2.0 * np.pi * k
    return np.sin(z/250.0) + np.sin(z/25.0)


def gen_input_signal_b_modified(k):
    """Generate sinusoidal input signal for the system."""
    z = 2.0 * np.pi * k
    return (np.sin(z/250.0) + np.sin(z/25.0)) / 2.0
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
            Sigmoidal())

        self.linear_stack.to(device)

    def forward(self, x):
        x = self.linear_stack(x)
        return x


def train_and_evaluate_fn1():
    print("# Non-linear Function 1")
    net = LearnedFn("cpu")
    criterion = nn.MSELoss()
    optim = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE)

    # Generate training data.
    # -----------------------------------------------,
    print(
        f"Generating {FN1_SIGA_EVAL_STEPS} training"
        f" samples using input signal 1 ...")
    time_steps = torch.arange(FN1_SIGA_EVAL_STEPS)
    y_list = []
    u_list = []
    f_list = []
    f_inp_list = []
    for k in time_steps:
        u_k = gen_input_signal_a(k)
        _, (f_k, f_k_inp), y_list, u_list = evolve_sys(
            y_list, u_list, calc_non_linear_fn1, u_k, k)

        f_list.append(f_k)
        f_inp_list.append(f_k_inp)
    # -----------------------------------------------'

    # Synchronously train and predict.
    # ------------------------------------------------,
    print(
        f"Training and evaluating network "
        f"for {FN1_SIGA_SYNC_TRAIN_STEPS} steps ...")
    pred_y_list = []
    pred_u_list = []

    # Convert data to pytorch tensors and add batch dimension.
    f_list = torch.from_numpy(
        np.array(f_list, dtype=np.float32).reshape(1, -1))
    f_inp_list = torch.from_numpy(
        np.array(f_inp_list, dtype=np.float32).reshape(1, -1))

    for k in torch.arange(FN1_SIGA_SYNC_TRAIN_STEPS):
        train_f = f_list[:, k]
        train_f_inp = f_inp_list[:, k]

        optim.zero_grad()
        pred_f = net(train_f_inp)
        loss = criterion(pred_f, train_f)
        loss.backward()
        optim.step()

        _, _, pred_y_list, pred_u_list = evolve_sys(
            pred_y_list,
            pred_u_list,
            lambda x: net(torch.tensor(x, dtype=torch.float32).reshape(1, 1)).item(),
            u_list[k],
            k)
    # ------------------------------------------------'

    # Keep predicting.
    # ------------------------------------------------,
    print("Evaluating network without lock-step training ...")
    for k in range(
            FN1_SIGA_SYNC_TRAIN_STEPS,
            FN1_SIGA_EVAL_STEPS):
        _, _, pred_y_list, pred_u_list = evolve_sys(
            pred_y_list,
            pred_u_list,
            lambda x: net(torch.tensor(x, dtype=torch.float32) .reshape(1, 1)).item(),
            u_list[k],
            k)
    # ------------------------------------------------'

    # Plot graph.
    print("Plotting graph ...")
    ax = plt.subplot()
    ax.set_title(
        "Ex. 1, $f(u) = 0.6sin(\pi u) + 0.3 sin(3 \pi u) + 0.1 sin (5 \pi u)$",
        fontsize=10)
    ax.plot(time_steps, y_list, label="y")
    ax.plot(time_steps, pred_y_list, label="$\^{y}$", linestyle="dashed")
    ax.axvline(FN1_SIGA_SYNC_TRAIN_STEPS, color="grey", linestyle="dotted")
    ax.set_ylabel("y vs $\^{y}$")
    ax.set_xlabel("Timestep, $k$")
    ax.legend()
    ax.figure.set_size_inches(8, 3)
    ax.figure.savefig(FN1_GRAPH_FILENAME)
    plt.tight_layout()
    plt.show()


def train_and_evaluate_fn2():
    print("# Non-linear Function 2")
    net = LearnedFn("cpu")
    criterion = nn.MSELoss()
    optim = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE)

    # Generate random training data.
    # -----------------------------------------------,
    print(
        f"Generating {FN2_SIGA_RANDOM_TRAIN_STEPS} "
        f"random training samples ...")
    # Choose some random input samples within (-1, 1).
    train_u_list = list(2.0 * (np.random.rand(FN2_SIGA_RANDOM_TRAIN_STEPS) - 0.5))
    train_y_list = []
    train_f_list = []
    train_f_inp_list = []
    for k, u_k in enumerate(train_u_list):
        _, (f_k, f_k_inp), train_y_list, train_u_list = evolve_sys(
            train_y_list,
            train_u_list,
            calc_non_linear_fn2,
            u_k,
            k)

        train_f_list.append(f_k)
        train_f_inp_list.append(f_k_inp)
    # -----------------------------------------------'

    # Train the network using the random training data.
    # ------------------------------------------------,
    print("Training network ...")
    # Convert data to pytorch tensors and add batch dimension.
    train_f_list = torch.from_numpy(
        np.array(train_f_list, dtype=np.float32).reshape(1, -1))
    train_f_inp_list = torch.from_numpy(
        np.array(train_f_inp_list, dtype=np.float32).reshape(1, -1))

    for k in range(FN2_SIGA_RANDOM_TRAIN_STEPS):
        train_f = train_f_list[:, k]
        train_f_inp = train_f_inp_list[:, k]

        optim.zero_grad()
        pred_f = net(train_f_inp)
        loss = criterion(pred_f, train_f)
        loss.backward()
        optim.step()
    # ------------------------------------------------'

    # Predict using input signal 2.
    # ------------------------------------------------,
    print("Evaluating identified model using combination of signal A and B ...")
    time_steps = range(FN2_SIGA_EVAL_STEPS + FN2_SIGB_EVAL_STEPS)

    # Purpose of this function is to generate two variants of graph using two
    # slightly different input signals.
    def eval_and_plot(
            gen_input_signal_b,
            title_suffix,
            graph_filename):

        def calc_u(k):
            return (
                gen_input_signal_a(k)
                if k < FN2_SIGA_EVAL_STEPS
                else gen_input_signal_b(k))

        # Generate ground truth data.
        eval_u_list = []
        eval_y_list = []
        for k in time_steps:
            u_k = calc_u(k)
            _, _, eval_y_list, eval_u_list = evolve_sys(
                eval_y_list,
                eval_u_list,
                calc_non_linear_fn2,
                u_k,
                k)

        # Generate prediction data.
        pred_u_list = []
        pred_y_list = []
        for k in time_steps:
            u_k = eval_u_list[k]
            _, _, pred_y_list, pred_u_list = evolve_sys(
                pred_y_list,
                pred_u_list,
                lambda x: net(torch.tensor(x, dtype=torch.float32).reshape(1, 1)).item(),
                u_k,
                k)

        # Plot graph.
        ax = plt.subplot()
        ax.set_title("Ex. 1, $f(u) = u^3 + 0.3u^2 - 0.4u$" + title_suffix)
        ax.plot(time_steps, eval_y_list, label="y")
        ax.plot(time_steps, pred_y_list, label="$\^{y}$", linestyle="dashed")
        ax.axvline(FN2_SIGA_EVAL_STEPS, color="grey", linestyle="dotted")
        ax.set_xlabel("Timestep, $k$")
        ax.set_ylabel("y vs $\^{y}$")
        ax.figure.set_size_inches(8, 3)
        ax.legend()
        ax.figure.tight_layout()
        ax.figure.savefig(graph_filename)
        plt.show()

    eval_and_plot(
        gen_input_signal_b,
        "; Original input signal",
        FN2_GRAPH1_FILENAME)

    eval_and_plot(
        gen_input_signal_b_modified,
        "; Second-half adjusted input signal",
        FN2_GRAPH2_FILENAME)
# ------------------------------------------------'


if __name__ == "__main__":
    train_and_evaluate_fn1()
    train_and_evaluate_fn2()
