"""This module reproduces example 7 from paper -

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
EVAL_STEPS = 100
FIG_MODEL_ID_FILENAME = "ex7_model_identification.pdf"
FIG_COMPARE_REF_VS_PLANT_NO_CTRL = "ex7_comparison_ref_model_vs_plant_no_control.pdf"
FIG_COMPARE_REF_VS_PLANT_WITH_CTRL = "ex7_comparison_ref_model_vs_plant_with_control.pdf"


# The initial conditions are chosen to be different than but within the
# equilibrium states of (0, 0) and (2, 2).
INIT_CONDITIONS = (0.5, 0.5)


def evolve_plant(yp_hist, u_hist, calc_f, u_k, k):
    """Evolve the dynamical system into time-step `k`.

    Args:
    - yp_hist: A list of past values of plant output `y` of size `(k - 1)`.
    - u_hist: A list of past values of plant input `u` of size `(k - 1)`.
    - calc_f: A non-linear function of plant outputs at `(k - 1)` and `(k - 2)`.
    - u_k: Plant input at time-step `k`.
    - k: The zero-indexed time-step.

    Returns:
    A tuple of format:
       `(yp_k, (f_k, f_k_inp1, f_k_inp2), yp_hist_updated, u_hist_updated)`
    where
      - `yp_k` is the plant output.
      - `f_k` is the output of `calc_f`.
      - `f_k_inp` is the input used to generate `f_k`.
      - `yp_hist_updated` is the updated `yp_hist`.
      - `u_hist_updated` is the updated `u_hist`.

    """
    f_k_inp1 = yp_hist[k - 1] if k > 0 else INIT_CONDITIONS[0]
    f_k_inp2 = yp_hist[k - 2] if k > 1 else INIT_CONDITIONS[1]
    f_k = calc_f(f_k_inp1, f_k_inp2)

    yp_k = f_k + u_hist[k - 1] if k > 0 else 0.0
    return (
        yp_k,
        (f_k, f_k_inp1, f_k_inp2),
        yp_hist + [yp_k],
        u_hist + [u_k])


def evolve_model(ym_hist, r_hist, r_k, k):
    """Evolve reference model into time-step `k`.

    Args:
    - ym_hist: A list of past values of model output `y` of size `(k - 1)`.
    - r_hist: A list of past values of model input `r` of size `(k - 1)`.
    - r_k: Model input at time-step `k`.

    Returns:
    A tuple of format:
       `(ym_k, yp_hist_updated, u_hist_updated)`
    where
      - `ym_k` is the model output.
      - `ym_hist_updated` is the updated `ym_hist`.
      - `r_hist_updated` is the updated `r_hist`.
    """
    past_y1 = ym_hist[k - 1] if k > 0 else INIT_CONDITIONS[0]
    past_y2 = ym_hist[k - 2] if k > 1 else INIT_CONDITIONS[1]
    past_rk = r_hist[k - 1] if k > 0 else 0.0

    ym_k = 0.6 * past_y1 + 0.2 * past_y2 + past_rk
    return (
        ym_k,
        ym_hist + [ym_k],
        r_hist + [r_k])


def calc_f(past_y1, past_y2):
    """Calculate ground truth of non-linear function `f`."""
    return (
        (past_y1 * past_y2 * (past_y1 + 2.5)) /
        (1.0 + np.pow(past_y1, 2) + np.pow(past_y2, 2)))


# Reference and control input signals.
# ------------------------------------------------------------,
def gen_ref_input_signal(k):
    """Generate reference input signal for the system."""
    z = 2 * np.pi * k
    return np.sin(z/25.0)


def gen_ctrl_input_signal(yp_hist, r_k, calc_f, k):
    """Generate control input signal for the system."""
    past_y1 = yp_hist[k - 1] if k > 0 else INIT_CONDITIONS[0]
    past_y2 = yp_hist[k - 2] if k > 1 else INIT_CONDITIONS[1]
    u_k = -calc_f(past_y1, past_y2) + 0.6 * past_y1 + 0.2 * past_y2 + r_k
    return u_k
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
            nn.Linear(2, 20),
            Sigmoidal(),
            nn.Linear(20, 10),
            Sigmoidal(),
            nn.Linear(10, 1))

        self.linear_stack.to(device)

    def forward(self, x):
        x = self.linear_stack(x)
        return x


def identify_model():
    net = LearnedFn("cpu")
    criterion = nn.MSELoss()
    optim = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE)

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
    for k, u_k in enumerate(training_input):
        _, (f_k, f_k_inp1, f_k_inp2), train_y_list, train_u_list = evolve_plant(
            train_y_list,
            train_u_list,
            calc_f,
            u_k,
            k)

        train_f_list.append(f_k)
        train_f_inp_list.append((f_k_inp1, f_k_inp2))
    # -----------------------------------------------'

    # Train network.
    # ------------------------------------------------,
    print("Training network ...")

    # Convert training data to pytorch tensors and add batch dimension.
    train_f_list = torch.from_numpy(
        np.array(train_f_list, dtype=np.float32)).reshape(1, -1)
    train_f_inp_list = torch.from_numpy(
        np.array(train_f_inp_list, dtype=np.float32)).reshape(1, -1, 2)

    for k in range(RANDOM_TRAIN_STEPS):
        train_f = train_f_list[:, k].unsqueeze(0)
        train_f_inp = train_f_inp_list[:, k, :]

        optim.zero_grad()
        pred_f = net(train_f_inp)
        loss = criterion(pred_f, train_f)
        loss.backward()

        optim.step()
    # ------------------------------------------------'

    # Test identified model.
    # ------------------------------------------------,
    print("Evaluating identified model ...")
    time_steps = range(EVAL_STEPS)

    # Generate test input and output signals.
    eval_r_list = []
    eval_u_list = []
    eval_y_list = []
    for k in time_steps:
        r_k = gen_ref_input_signal(k)
        _, _, eval_y_list, eval_u_list = evolve_plant(
            eval_y_list,
            eval_u_list,
            calc_f,
            r_k,
            k)
        eval_r_list += [r_k]

    # Predict output signal using the plant with learned network and control
    # input.
    pred_r_list = []
    pred_u_list = []
    pred_y_list = []
    for k in time_steps:
        r_k = gen_ref_input_signal(k)
        _, _, pred_y_list, pred_u_list = evolve_plant(
            pred_y_list,
            pred_u_list,
            lambda x1, x2: net(torch.tensor((x1, x2), dtype=torch.float32).reshape(1, 2)).item(),
            r_k,
            k)

        pred_r_list += [r_k]

    # Plot graph of true y vs predicted y.
    ax = plt.subplot()
    ax.set_title(
        "Example 7: Model Identification ; Input signal $u = sin(2 \pi k / 25)$",
        fontsize=10)
    ax.plot(time_steps, eval_y_list, label="y")
    ax.plot(time_steps, pred_y_list, label="$\^{y}$", linestyle="dashed")
    ax.set_xlabel("Timestep, $k$")
    ax.set_ylabel("y vs $\^{y}$")
    ax.figure.set_size_inches(8, 3)
    ax.legend()
    ax.figure.tight_layout()
    ax.figure.savefig(FIG_MODEL_ID_FILENAME)
    plt.show()

    return net


def compare_plant_and_model(
        net,
        control_on,
        graph_title,
        graph_fig_filename):
    print(
        f"Comparing plant and model "
        f"with control_on={control_on} ...")

    time_steps = range(EVAL_STEPS)

    # Generate reference input and reference output.
    r_list = []
    ym_list = []
    for k in time_steps:
        r_k = gen_ref_input_signal(k)
        _, ym_list, r_list = evolve_model(
            ym_list,
            r_list,
            r_k,
            k)

    # Generate plant output without any control.
    u_list = []
    yp_list = []
    for k in time_steps:
        if control_on:
            # Plant input is generated by a control function which uses the
            # reference input and modifies it using the trained network.
            u_k = gen_ctrl_input_signal(
                yp_list,
                r_list[k],
                lambda x1, x2: net(
                    torch.tensor(
                        (x1, x2),
                        dtype=torch.float32).reshape(1, 2)).item(),
                k)
        else:
            # Plant input is same as reference input, r_k, since there is no
            # control.
            u_k = r_list[k]

        _, _, yp_list, u_list = evolve_plant(
            yp_list,
            u_list,
            calc_f,
            u_k,
            k)

    # Plot graph of true y vs predicted y.
    ax = plt.subplot()
    ax.set_title(graph_title, fontsize=10)
    ax.plot(time_steps, ym_list, label="$y_m$")
    ax.plot(time_steps, yp_list, label="$y_p$", linestyle="dashed")
    ax.set_xlabel("Timestep, $k$")
    ax.set_ylabel("$y_m$ vs $y_p$")
    ax.figure.set_size_inches(8, 3)
    ax.legend()
    ax.figure.tight_layout()
    ax.figure.savefig(graph_fig_filename)
    plt.show()


if __name__ == "__main__":
    net = identify_model()
    compare_plant_and_model(
        net,
        control_on=False,
        graph_title="Example 7: Reference model vs no-control plant; Input $r = sin(2 \pi k / 25)$",
        graph_fig_filename=FIG_COMPARE_REF_VS_PLANT_NO_CTRL)
    compare_plant_and_model(
        net,
        control_on=True,
        graph_title="Example 7: Reference model vs controlled plant; Input $r = sin(2 \pi k / 25)$",
        graph_fig_filename=FIG_COMPARE_REF_VS_PLANT_WITH_CTRL)
