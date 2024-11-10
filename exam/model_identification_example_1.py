"""This module reproduces example 1 from paper -

Identification and Control of Dynamical Systems Using Neural Networks
KUMPATI S. NARENDRA FELLOW, IEEE. AND KANNAN PARTHASARATHY

Author: Titon Barua <baruat@email.sc.edu>
"""

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


def evolve_sys(y_hist, non_linear_fn_output):
    """Calculate output of the dynamical system."""
    past_y1 = y_hist[-1] if len(y_hist) > 0 else 0.0
    past_y2 = y_hist[-2] if len(y_hist) > 1 else 0.0
    return 0.3 * past_y1 + 0.6 * past_y2 + non_linear_fn_output


def calc_non_linear_fn1(u):
    """Calculate ground truth of non-linear function."""
    pi_u = torch.pi * u
    return (
        0.6 * torch.sin(pi_u) +
        0.3 * torch.sin(3 * pi_u) +
        0.1 * torch.sin(5 * pi_u))


def calc_non_linear_fn2(u):
    """Calculate ground truth of non-linear function."""
    return (u * u * u) + 0.3 * (u * u) - 0.4 * u


# Input signals used for training and evaluation.
# ------------------------------------------------------------,
def gen_input_signal_a(k):
    """Generate sinusoidal input signal for the system."""
    return torch.sin((2.0 * torch.pi * k) / 250.0)


def gen_input_signal_b(k):
    """Generate sinusoidal input signal for the system."""
    z = 2.0 * torch.pi * k
    return torch.sin(z/250.0) + torch.sin(z/25.0)


def gen_input_signal_b_modified(k):
    """Generate sinusoidal input signal for the system."""
    z = 2.0 * torch.pi * k
    return (torch.sin(z/250.0) + torch.sin(z/25.0)) / 2.0
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
    time_steps = torch.arange(0, FN1_SIGA_EVAL_STEPS)
    x_list = []
    z_list = []
    y_list = []
    for k in time_steps:
        x = gen_input_signal_a(k)
        z = calc_non_linear_fn1(x)
        y = evolve_sys(y_list, z)
        x_list.append(x)
        z_list.append(z)
        y_list.append(y)
    # -----------------------------------------------'

    # Synchronously train and predict.
    # ------------------------------------------------,
    print(
        f"Training and evaluating network "
        f"for {FN1_SIGA_SYNC_TRAIN_STEPS} steps ...")
    pred_y_list = []
    for k in range(FN1_SIGA_SYNC_TRAIN_STEPS):
        train_x = torch.tensor(x_list[k]).unsqueeze(0)
        train_z = torch.tensor(z_list[k]).unsqueeze(0)

        optim.zero_grad()
        pred_z = net(train_x)
        loss = criterion(pred_z, train_z)
        loss.backward()
        optim.step()

        pred_y_list.append(
           evolve_sys(
               pred_y_list,
               net(train_x).item()))
    # ------------------------------------------------'

    # Keep predicting.
    # ------------------------------------------------,
    print("Evaluating network without lock-step training ...")
    for k in range(
            FN1_SIGA_SYNC_TRAIN_STEPS,
            FN1_SIGA_EVAL_STEPS):
        eval_x = torch.tensor(x_list[k]).unsqueeze(0)
        pred_y_list.append(
            evolve_sys(
                pred_y_list,
                net(eval_x).item()))
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
    train_x_list = 2.0 * (torch.rand(FN2_SIGA_RANDOM_TRAIN_STEPS) - 0.5)
    train_z_list = []
    train_y_list = []
    for x in train_x_list:
        z = calc_non_linear_fn2(x)
        y = evolve_sys(train_y_list, z)
        train_z_list.append(z)
        train_y_list.append(y)
    # -----------------------------------------------'

    # Train the network using the random training data.
    # ------------------------------------------------,
    print("Training network ...")
    for k in range(FN2_SIGA_RANDOM_TRAIN_STEPS):
        train_x = train_x_list[k].unsqueeze(0)
        train_z = train_z_list[k].unsqueeze(0)

        optim.zero_grad()
        pred_z = net(train_x)
        loss = criterion(pred_z, train_z)
        loss.backward()
        optim.step()
    # ------------------------------------------------'

    # Predict using input signal 2.
    # ------------------------------------------------,
    print("Evaluating identified model using combination of signal A and B ...")
    time_steps = torch.arange(
        FN2_SIGA_EVAL_STEPS + FN2_SIGB_EVAL_STEPS)

    def eval_and_plot(
            gen_input_signal_b,
            title_suffix,
            graph_filename):
        eval_x_list = []
        eval_y_list = []
        for k in time_steps:
            x = (gen_input_signal_a(k)
                 if k < FN2_SIGA_EVAL_STEPS
                 else gen_input_signal_b(k))
            z = calc_non_linear_fn2(x)
            y = evolve_sys(eval_y_list, z)
            eval_x_list.append(x)
            eval_y_list.append(y)

        pred_y_list = []
        for k in time_steps:
            eval_x = eval_x_list[k].unsqueeze(0)
            pred_y_list.append(
                evolve_sys(
                    pred_y_list,
                    net(eval_x).item()))

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
