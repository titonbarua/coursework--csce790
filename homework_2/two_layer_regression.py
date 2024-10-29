import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error
from network import MLP, train_network


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
        {"type": "linear", "n_neurons": 1000},
        {"type": "relu"},
        {"type": "linear", "n_neurons": 1},
    ])


epoch_vs_mae_error = []
epoch_vs_predictions = []


def snapshot_callback(network, props):
    epoch = props["epoch"]

    pred_Y = network.infer_batch(train_X)
    if epoch in (10, 100, 200, 400, 1000):
        epoch_vs_predictions.append((epoch, pred_Y))

    mae_error = mean_absolute_error(train_Y, pred_Y)
    epoch_vs_mae_error.append((epoch, mae_error))
    print("Epoch: {}, MAE: {}".format(epoch, mae_error))


train_network(
    net,
    train_X,
    train_Y,
    calc_accuracy_fn=r2_score,
    snapshot_callback=snapshot_callback,
    print_progress=False,
    learning_rate=0.001)
