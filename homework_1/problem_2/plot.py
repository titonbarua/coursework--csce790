import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from functools import partial
from matplotlib import cm


PREVIEW = False
N_SAMPLES_LIST = [100, 5000, 10000]
HARD_LIMIT_THRESHOLD = 0.0
GAUSSIAN_RBF_EPSILON = 1.0


try:
    if not PREVIEW:
        matplotlib.use("pdf")
except ValueError:
    pass


def sigmoid(x):
    """Sigmoid function."""
    return 1.0 / (1.0 + np.exp(-x))


def hard_limit(x):
    """A hard_limit function with `limit` as the limit."""
    return 0.0 if x < HARD_LIMIT_THRESHOLD else 1.0


def gaussian_radial_basis_func(x):
    """A radial basis function with `epsilon` as the shape parameter."""
    return np.exp(-np.square(GAUSSIAN_RBF_EPSILON * x))


def create_random_input_samples(
        n_samples,
        x1_start=-2.0,
        x1_end=2.0,
        x2_start=-2.0,
        x2_end=2.0):
    """Randomly choose some samples in a 2D domain."""
    x12 = np.random.uniform(
        low=[x1_start, x2_start],
        high=[x1_end, x2_end],
        size=(n_samples, 2))

    return x12


def create_input_samples_grid(
        n_samples,
        x1_start=-2.0,
        x1_end=2.0,
        x2_start=-2.0,
        x2_end=2.0):
    """Choose some samples in a 2D domain in a grid-like fashion."""
    m = math.ceil(math.sqrt(n_samples))
    x1 = np.linspace(x1_start, x1_end, num=m)
    x2 = np.linspace(x2_start, x2_end, num=m)
    x1, x2 = np.meshgrid(x1, x2, indexing='xy')
    return np.stack((x1, x2), axis=-1)


def perceptron1(activation_func, x):
    assert x.shape == (2,)
    yi = -4.79 * x[0] + 5.90 * x[1] - 0.93
    return activation_func(yi)


perceptron1_sigmoid = partial(
    perceptron1,
    sigmoid)
perceptron1_hard_limit = partial(
    perceptron1,
    hard_limit)
perceptron1_gaussian_rbf = partial(
    perceptron1,
    gaussian_radial_basis_func)


PC2_VT = np.array([-2.69, -2.80, -3.39, -4.56]).reshape(2, 2)
PC2_bv = np.array([-2.21, 4.76]).reshape(2, 1)
PC2_W = np.array([-4.91, 4.95]).reshape(2, 1)
PC2_bw = -2.28


def perceptron2(numpy_activation_func, x):
    assert x.shape == (2,)
    Yi = PC2_VT @ x.reshape(2, 1) + PC2_bv
    y = PC2_W.T @ numpy_activation_func(Yi) + PC2_bw
    return y.item()


perceptron2_sigmoid = partial(
    perceptron2,
    np.frompyfunc(sigmoid, 1, 1))
perceptron2_hard_limit = partial(
    perceptron2,
    np.frompyfunc(hard_limit, 1, 1))
perceptron2_gaussian_rbf = partial(
    perceptron2,
    np.frompyfunc(gaussian_radial_basis_func, 1, 1))


def plot_perceptrons(
        n_samples_list,
        perceptron_list,
        perceptron_names,
        view_angle_list,
        save_path=None,
        figure_title=None):
    fig, axes = plt.subplots(
        len(n_samples_list),
        len(perceptron_list),
        subplot_kw={"projection": "3d",
                    "proj_type": "ortho"},
        figsize=(6.5, 8.5))

    if figure_title:
        fig.suptitle(figure_title, fontweight="bold")

    for i, n_samples in enumerate(n_samples_list):
        x = create_input_samples_grid(n_samples)
        for j, (perceptron, perceptron_name, view_angle) in enumerate(
                zip(perceptron_list,
                    perceptron_names,
                    view_angle_list)):
            y = np.apply_along_axis(perceptron, axis=2, arr=x)
            # print(y.shape)

            ax = axes[i, j]
            ax.plot_surface(
                x[:, :, 0],
                x[:, :, 1],
                y,
                linewidth=0,
                cmap=cm.viridis,
                rcount=np.sqrt(n_samples) * 10,
                ccount=np.sqrt(n_samples) * 10,
                antialiased=True)

            ax.view_init(*view_angle)
            ax.set_title(
                perceptron_name + "\nn={}".format(y.size),
                fontsize=10,
                pad=5)

            ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1.0))
            ax.xaxis.set_major_formatter("{x:0.1f}")
            ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.5))
            ax.xaxis.set_minor_formatter("")
            ax.xaxis.labelpad = -5
            ax.set_xlim(-2.01, 2.01)
            ax.set_xlabel(r"$x1$", fontsize=8, color="red")
            ax.tick_params(
                axis='x', which='both', labelcolor="red", labelsize=5, pad=-2)

            ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1.0))
            ax.yaxis.set_major_formatter("{x:0.1f}")
            ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.5))
            ax.yaxis.set_minor_formatter("")
            ax.yaxis.labelpad = -5
            ax.set_ylim(-2.0, 2.0)
            ax.set_ylabel(r"$x2$", fontsize=8, color="green")
            ax.tick_params(
                axis='y', which='both', labelcolor="green", labelsize=5, pad=-2)

            ax.zaxis.set_major_locator(matplotlib.ticker.AutoLocator())
            ax.zaxis.set_major_formatter("{x:0.1f}")
            ax.zaxis.labelpad = -5
            ax.set_zlabel(r"$y$", fontsize=8, color="blue")
            ax.tick_params(
                axis='z', which='both', labelcolor="blue", labelsize=5, pad=-2)


    plt.subplots_adjust(top=1.0)
    plt.tight_layout(h_pad=3.0, w_pad=2.0)

    if PREVIEW:
        plt.show()
    else:
        if save_path:
            plt.savefig(save_path)


if __name__ == "__main__":
    print("Plotting first perceptron variants ...")
    plot_perceptrons(
        N_SAMPLES_LIST,
        [perceptron1_sigmoid,
         perceptron1_hard_limit,
         perceptron1_gaussian_rbf],
        ["Sigmoid",
         r"Hard Limit $(t=0.0)$",
         r"Gaussian RBF ($\epsilon = 1.0$)"],
        [[20, 30, 0],
         [20, 30, 0],
         [20, 30, 0]],
        figure_title=r"Surface Plots of $y = \sigma(-4.79x_1 + 5.90x_2 - 0.93)$",
        save_path="perceptron_graph_a.pdf")

    print("Plotting second perceptron variants ...")
    plot_perceptrons(
        N_SAMPLES_LIST,
        [perceptron2_sigmoid,
         perceptron2_hard_limit,
         perceptron2_gaussian_rbf],
        ["Sigmoid",
         r"Hard Limit $(t=0.0)$",
         r"Gaussian RBF ($\epsilon = 1.0$)"],
        [[30, 116, 0],
         [30, 116, 0],
         [40, 130, 0]],
        figure_title=r"Surface plots of $y = W^T\sigma(V^Tx + b_v) + b_w$",
        save_path="perceptron_graph_b.pdf")
