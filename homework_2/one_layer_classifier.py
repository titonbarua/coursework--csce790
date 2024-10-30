import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from network import (
    MLP,
    mse_loss,
    bce_loss,
    grad_mse_loss,
    grad_bce_loss,
    train_network)


training_X = np.array(
    [0.1, 1.2,
     0.7, 1.8,
     0.8, 1.6,
     0.8, 0.6,
     1.0, 0.8,
     0.3, 0.5,
     0.0, 0.2,
     -0.3, 0.8,
     -0.5, -1.5,
     -1.5, -1.3],
    dtype=np.float64).reshape(-1, 2)


training_Y = np.array(
    [1, 0,
     1, 0,
     1, 0,
     0, 0,
     0, 0,
     1, 1,
     1, 1,
     1, 1,
     0, 1,
     0, 1],
    dtype=np.float64).reshape(-1, 2)


def thresholded_accuracy_score(Y_true, Y_pred, threshold=0.5):
    """Calculate accuracy store with `Y_pred` converted to binary using a
    threshold.

    """
    Y_pred_bin = np.zeros_like(Y_pred, dtype=np.int8)
    Y_pred_bin[Y_pred > threshold] = 1
    return accuracy_score(Y_true, Y_pred_bin)


net = MLP(
    input_dim=2,
    layer_spec=[
        {"type": "linear", "n_neurons": 120},
        {"type": "sigmoid"},
        {"type": "linear", "n_neurons": 2},
        {"type": "sigmoid"},
    ],
    loss_fn=bce_loss,
    loss_grad_fn=grad_bce_loss,
    debug=False)


def plot_epoch_vs_bce_error_graph(
        epoch_vs_bce,
        epoch_vs_accuracy,
        filepath=None):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    fig.suptitle("Training Epoch vs BCE Error Graph")

    epochs = [ep for ep, _ in epoch_vs_bce]
    bce_y1 = [bce[0] for _, bce in epoch_vs_bce]
    bce_y2 = [bce[1] for _, bce in epoch_vs_bce]
    acc = [x for _, x in epoch_vs_accuracy]

    ax1.plot(epochs, bce_y1, color="blue", label="y1")
    ax1.plot(epochs, bce_y2, color="red", label="y2")
    ax1.grid()
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("BCE Error")

    ax2.plot(epochs, acc, color="green", linestyle="--", label="accuracy")
    ax2.set_ylabel("Accuracy")

    ax2.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0 - 0.1))
    ax1.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0 - 0.2))

    if filepath:
        fig.savefig(filepath, dpi=300)

    plt.show(block=True)


DB_AXIS_POINTS = 100
decision_boundary_grid = np.meshgrid(
    np.linspace(-2.0, 2.0, DB_AXIS_POINTS),
    np.linspace(-2.0, 2.0, DB_AXIS_POINTS))


def plot_decision_boundaries(epoch_vs_decision_boundary, filepath=None):
    rows = len(epoch_vs_decision_boundary)
    cols = 2

    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * 3 + 0.25, rows * 3 + 0.25 * 2))
    fig.suptitle("Decision Boundaries", fontsize=10)

    for i in range(rows):
        epoch, X, Y = epoch_vs_decision_boundary[i]
        GRID_X1 = X[:, 0].reshape(DB_AXIS_POINTS, DB_AXIS_POINTS)
        GRID_X2 = X[:, 1].reshape(DB_AXIS_POINTS, DB_AXIS_POINTS)

        TX1 = training_X[:, 0]
        TX2 = training_X[:, 1]

        for j in range(cols):
            GRID_Z = Y[:, j].reshape(DB_AXIS_POINTS, DB_AXIS_POINTS)
            ax = axes[i][j]
            ax.set_aspect('equal')
            ax.set_xlabel("x1")
            ax.set_ylabel("x2")
            ax.set_title(
                "epoch: {}, var: y{}"
                .format(epoch, j + 1), fontsize=8)

            # Fill different decision areas and draw boundary.
            ax.contourf(
                GRID_X1, GRID_X2, GRID_Z,
                colors=["tomato", "chartreuse", "chartreuse"],
                levels=[0.0, 0.50, 1.0],
                alpha=0.5)
            ax.contour(
                GRID_X1, GRID_X2, GRID_Z,
                colors=["black"],
                levels=[0.5],
                alpha=0.75)

            TY = training_Y[:, j]
            TY_pos = np.argwhere(TY == 1)
            TY_neg = np.argwhere(TY == 0)
            ax.scatter(
                TX1[TY_pos], TX2[TY_pos],
                color="green", marker="^", label="1")
            ax.scatter(
                TX1[TY_neg], TX2[TY_neg],
                color="red", marker="x", label="0")
            ax.legend(loc="lower right")

    fig.tight_layout()

    if filepath:
        fig.savefig(filepath)

    plt.show()


epoch_vs_bce_error = []
epoch_vs_accuracy = []
epoch_vs_decision_boundary = []


def snapshot_callback(network, props):
    epoch = props["epoch"]
    pred_Y = network.infer_batch(training_X).squeeze()
    bce_error = bce_loss(training_Y, pred_Y)
    epoch_vs_bce_error.append((epoch, bce_error))
    epoch_vs_accuracy.append((epoch, props["accuracy"]))

    if epoch in (3, 10, 100):
        x1v = decision_boundary_grid[0].reshape(-1, 1)
        x2v = decision_boundary_grid[1].reshape(-1, 1)
        X = np.hstack((x1v, x2v))
        Y = network.infer_batch(X)
        epoch_vs_decision_boundary.append((epoch, X, Y))



train_network(
    net,
    training_X,
    training_Y,
    epochs=100,
    snapshot_callback=snapshot_callback,
    calc_accuracy_fn=thresholded_accuracy_score)


plot_epoch_vs_bce_error_graph(
    epoch_vs_bce_error,
    epoch_vs_accuracy,
    "b.1-epoch_vs_bce_error.pdf")

plot_decision_boundaries(
    epoch_vs_decision_boundary,
    "b.1-decision_boundaries.pdf")
