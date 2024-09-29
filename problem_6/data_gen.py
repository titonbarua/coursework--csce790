import os
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from functools import partial


PREVIEW = False

if not PREVIEW:
    matplotlib.use("pdf")


def calc_derivative_rossler(a, b, c, state):
    # assert state.shape == (3, 1)
    x, y, z = state

    dx_dt = -y - z
    dy_dt = x + a * y
    dz_dt = b + z * (x - c)

    return np.array([dx_dt, dy_dt, dz_dt]).reshape(3, 1)


def calc_derivative_lorenz(a, b, c, state):
    # assert state.shape == (3, 1)
    x, y, z = state

    dx_dt = a * (y - x)
    dy_dt = (b - z) * x - y
    dz_dt = x * y - c * z  # The equation in the paper is wrong!!

    return np.array([dx_dt, dy_dt, dz_dt]).reshape(3, 1)


def normalize_time_series(data):
    """Normalize data using eqn. 14 from the paper."""
    n_data = data.shape[0]
    outp = data[:]
    states = data[:, 1:]

    for i in range(1, n_data):
        mean = np.mean(states[:i+1, :], axis=0)
        p = states[i, :] - mean
        q = np.sqrt(np.mean(np.square(states[:i+1, :] - mean)))
        outp[i, 1:] = p / q

    return outp


def solve_ode_with_euler_approx(
        calc_derivative_func,
        init_state,
        total_time,
        time_interval,
        euler_steps_per_interval):
    """Generate solution for an ODE using first order euler approximation."""
    init_state = np.array(init_state, dtype=np.float128).reshape(-1, 1)
    n_steps = int(total_time / time_interval)
    h = time_interval / euler_steps_per_interval

    t = [0.0]
    data = [init_state]

    state = init_state
    for i in range(n_steps):
        for _ in range(euler_steps_per_interval):
            dv = calc_derivative_func(state)
            state = state + dv * h
        t.append(i * time_interval)
        data.append(state)

    return np.hstack((
        np.array(t).reshape(-1, 1),
        np.array(data).reshape(-1, init_state.shape[0])))


def plot_data(
        data,
        title,
        savepath,
        axis_labels=[r"$x(t)$", r"$y(t)$", r"$z(t)$"]):
    fig, axes = plt.subplots(3, 1, figsize=(8, 8))
    fig.suptitle(title)

    axes[0].plot(data[:, 0], data[:, 1])
    axes[0].set_ylabel(axis_labels[0])
    axes[1].plot(data[:, 0], data[:, 2])
    axes[1].set_ylabel(axis_labels[1])
    axes[2].plot(data[:, 0], data[:, 3])
    axes[2].set_ylabel(axis_labels[2])

    if PREVIEW:
        plt.show()
    else:
        plt.savefig(savepath)


def main():
    os.makedirs("data/", exist_ok=True)
    os.makedirs("graphs/", exist_ok=True)

    print("Generating rossler system data ...")
    rossler_raw_data = solve_ode_with_euler_approx(
        partial(calc_derivative_rossler, 0.5, 2.0, 4.0),
        init_state=[0.0, 0.0, 0.0],
        total_time=260 + 100,
        time_interval=0.1,
        euler_steps_per_interval=1000)

    rossler_norm_data = normalize_time_series(rossler_raw_data)

    plot_data(rossler_raw_data, "Rossler Raw Data", "graphs/rossler_raw.pdf")
    plot_data(rossler_norm_data, "Rossler Normalized Data", "graphs/rossler_norm.pdf")

    np.savetxt("data/rossler_raw.nptxt", rossler_raw_data, header="# t x y z")
    np.savetxt("data/rossler_norm.nptxt", rossler_norm_data, header="# t x y z")

    print("Generating lorenz system data ...")
    lorenz_raw_data = solve_ode_with_euler_approx(
        partial(calc_derivative_lorenz, 10.0, 28.0, 8.0/3.0),
        init_state=[0.9, 0.0, 0.0],
        total_time=36,
        time_interval=0.05,
        euler_steps_per_interval=1000)

    lorenz_norm_data = normalize_time_series(lorenz_raw_data)

    lorenz_sqr_raw_data = lorenz_raw_data[:]
    lorenz_sqr_raw_data[:, 1:] = np.square(lorenz_sqr_raw_data[:, 1:])
    lorenz_sqr_norm_data = normalize_time_series(lorenz_sqr_raw_data)

    plot_data(lorenz_raw_data, "Lorenz Raw Data", "graphs/lorenz_raw.pdf")
    plot_data(lorenz_norm_data, "Lorenz Normalized Data", "graphs/lorenz_norm.pdf")
    plot_data(lorenz_sqr_norm_data, "Lorenz Square Normalized Data", "graphs/lorenz_sqr_norm.pdf")

    np.savetxt("data/lorenz_raw.nptxt", lorenz_raw_data, header="# t x y z")
    np.savetxt("data/lorenz_norm.nptxt", lorenz_norm_data, header="# t x y z")
    np.savetxt("data/lorenz_sqr_norm.nptxt", lorenz_sqr_norm_data, header="# t x y z")


if __name__ == "__main__":
    main()
