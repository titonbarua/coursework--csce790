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


epoch_vs_bce_error = []
epoch_vs_accuracy = []


def snapshot_callback(network, props):
    epoch = props["epoch"]
    pred_Y = network.infer_batch(training_X).squeeze()
    bce_error = bce_loss(training_Y, pred_Y)
    epoch_vs_bce_error.append((epoch, bce_error))
    epoch_vs_accuracy.append((epoch, props["accuracy"]))


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
    "2.a-epoch_vs_bce_error.pdf")
