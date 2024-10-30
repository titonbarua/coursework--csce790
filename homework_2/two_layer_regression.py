import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from network import MLP, train_network
import matplotlib as mpl
import matplotlib.pyplot as plt


train_X = np.array(
    [-1.0,
     -0.9,
     -0.8,
     -0.7,
     -0.6,
     -0.5,
     -0.4,
     -0.3,
     -0.2,
     -0.1,
     0.0,
     0.1,
     0.2,
     0.3,
     0.4,
     0.5,
     0.6,
     0.7,
     0.8,
     0.9,
     1.0])

train_Y = np.array(
    [-0.96,
     -0.577,
     -0.073,
     0.377,
     0.641,
     0.66,
     0.461,
     0.134,
     -0.201,
     -0.434,
     -0.5,
     -0.393,
     -0.165,
     0.099,
     0.307,
     0.396,
     0.345,
     0.182,
     -0.031,
     -0.219,
     -0.321])


net = MLP(
    input_dim=1,
    layer_spec=[
        {"type": "linear", "n_neurons": 35},
        {"type": "relu"},
        {"type": "linear", "n_neurons": 20},
        {"type": "relu"},
        {"type": "linear", "n_neurons": 1},
    ])


def plot_epoch_vs_mse_error_graph(epoch_vs_mse, filepath=None):
    fig, ax = plt.subplots()

    epochs = [ep for ep, _ in epoch_vs_mse]
    mse_errors = [err for _, err in epoch_vs_mse]

    ax.plot(epochs, mse_errors, color="blue")
    ax.grid()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Mean Squared Error")

    fig.suptitle("Training Epoch vs MSE Error Graph")

    if filepath:
        fig.savefig(filepath, dpi=300)

    plt.show(block=True)


def plot_predictions(
        train_X,
        train_Y,
        epoch_vs_pred,
        filepath=None):
    mpl.style.use("default")
    fig, ax = plt.subplots()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.suptitle("Comparison of predictions from different training epochs")

    # Draw ground truth.
    ax.scatter(train_X, train_Y, color="gray", label='$f(x)$', marker='+')

    # Draw neural network outputs by epoch.
    for i, (epoch, pred) in enumerate(epoch_vs_pred):
        ax.plot(train_X, pred, color=f'C{i}', label=f"epoch {epoch}", alpha=0.7)

    ax.legend()

    if filepath:
        fig.savefig(filepath, dpi=300)

    plt.show(block=True)


epoch_vs_mse_error = []
epoch_vs_predictions = []


def snapshot_callback(network, props):
    epoch = props["epoch"]

    pred_Y = network.infer_batch(train_X)
    if epoch in (10, 100, 200, 400, 1000):
        epoch_vs_predictions.append((epoch, pred_Y))

    mse_error = mean_squared_error(train_Y, pred_Y)
    epoch_vs_mse_error.append((epoch, mse_error))
    # print("Epoch: {}, MSE: {}".format(epoch, mse_error))


train_network(
    net,
    train_X,
    train_Y,
    calc_accuracy_fn=r2_score,
    snapshot_callback=snapshot_callback,
    print_progress=False,
    epochs=1000,
    learning_rate=0.001)


plot_epoch_vs_mse_error_graph(
    epoch_vs_mse_error,
    "b.2-epoch_vs_mse_error_graph.pdf")

plot_predictions(
    train_X,
    train_Y,
    epoch_vs_predictions,
    filepath="b.2-prediction_comparison.pdf")
