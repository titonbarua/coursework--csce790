import numpy as np
from sklearn.metrics import r2_score

rng = np.random.default_rng()


def relu(X):
    """ReLU activation function."""
    return np.where(X > 0.0, X, 0.0)


def grad_relu(X):
    """Calculate gradient of ReLU w.r.t. to it's input."""
    return np.where(X <= 0.0, 0.0, 1.0)


def sigmoid(X):
    """Sigmoid activation function."""
    # We are using this formulation to prevent overflows and underflows.
    # Ref: https://stackoverflow.com/a/51976485
    return np.where(
        X >= 0,
        1.0 / (1.0 + np.exp(-X)),
        np.exp(X) / (1.0 + np.exp(X)))


def grad_sigmoid(X):
    """Calculated gradient of sigmoid w.r.t it's input."""
    s = sigmoid(X)
    return s * (1.0 - s)


def mse_loss(Y_target, Y_pred):
    """Mean-square-error loss."""
    return np.mean(np.square(Y_target - Y_pred), axis=1)


def grad_mse_loss(Y_target, Y_pred):
    Y_target = Y_target.reshape(-1, 1)
    Y_pred = Y_pred.reshape(-1, 1)
    return 2 * (Y_pred - Y_target)


def bce_loss(Y_target, Y_pred):
    # https://sebastianraschka.com/blog/2022/losses-learned-part1.html
    first = -Y_target @ np.log(Y_pred)
    second = -(1 - Y_target) @ np.log(1 - Y_pred)
    return (first + second) / Y_target.shape[0]


def grad_bce_loss(Y_target, Y_pred):
    # https://math.stackexchange.com/a/3220477
    Y_target = Y_target.reshape(-1, 1)
    Y_pred = Y_pred.reshape(-1, 1)
    a = Y_pred - Y_target
    b = Y_pred - Y_pred * Y_pred
    return a / b


def uniform_initializer(W, low=-1.0, high=1.0):
    """Initialize the weights uniformly within a range."""
    return rng.uniform(low, high, W.shape)


def print_signals(signals):
    """Print the signals step by step in human readable format."""
    print("#" + "-" * 75 + ",")
    for i, (layer_index, layer_type, arr) in enumerate(signals):
        print(f"{layer_index}: {layer_type}")
        print(arr)
        if i < len(signals) - 1:
            print("-" * 50)
    print("#" + "-" * 75 + "'")


class MLP(object):
    """Class to implement a multi-level-perceptron."""

    def __init__(
            self,
            input_dim,
            layer_spec,
            loss_fn=mse_loss,
            loss_grad_fn=grad_mse_loss,
            init_fn=uniform_initializer,
            debug=False):
        """Initialize an MLP.

        Args:
        - input_dim: Number of input scalars.
        - layer_spec: Specification of the layers. It is a list of tuples of
          format: (<layer-type>, <n-neurons>).
        - init_fn(optional): Function to initialize the linear layer weights.

        """
        self.input_dim = input_dim
        self.layer_spec = layer_spec
        self.init_fn = init_fn
        self.loss_fn = loss_fn
        self.loss_grad_fn = loss_grad_fn
        self.debug = debug

        self.weights = []

        hidden_layer_sizes = [
            ls['n_neurons'] for ls in layer_spec
            if ls['type'] == 'linear']

        # All the input sizes are increased by 1 to incorporate bias.
        inp_sizes = [(s + 1) for s in ([input_dim] + hidden_layer_sizes[:-1])]
        outp_sizes = hidden_layer_sizes

        # Initialize linear layer weights.
        for inp_s, outp_s in zip(inp_sizes, outp_sizes):
            W = np.zeros((outp_s, inp_s), dtype=np.float64)
            if init_fn:
                W = uniform_initializer(W)
            self.weights.append(W)

        # for w in self.weights:
        #   print(w.shape)

    def get_loss(self, Y_target, signals):
        """Calculate loss by comparing prediction vs target label."""
        Y_pred = signals[-1][-1]
        loss = self.loss_fn(Y_target, Y_pred)
        return loss

    def forward_pass(self, input_):
        """Pass the input signal through the network and generate output.

        Returns a list of signals, last of which is the output of the network
        and the rest are the intermediate computations at each layer.

        """
        # Change input array into (N, 1) shape.
        input_ = np.array(input_).reshape(-1, 1)
        assert input_.shape[0] == self.input_dim

        X = input_[:]
        signals = [(0, 'input', X)]

        i_linear = 0
        for layer_index, ls in enumerate(self.layer_spec):
            ltype = ls["type"]
            if ltype == 'linear':
                Wb = self.weights[i_linear]

                # Append constant input of 1.0 at the end, to account for bias.
                X_biased = np.append(X, [[1.0]], axis=0)
                X = Wb @ X_biased
                i_linear += 1

            elif ltype == 'relu':
                X = relu(X)

            elif ltype == 'sigmoid':
                X = sigmoid(X)

            else:
                raise ValueError(f"Unknown layer type: {ltype}!")

            signals.append((layer_index + 1, ltype, X))

        return signals

    def infer(self, input_):
        """Make a prediction from the given input."""
        return self.forward_pass(input_)[-1][-1]

    def infer_batch(self, input_batch):
        """Predict a batch of input data."""
        Y_batch = []
        for X in input_batch:
            Y_batch.append(self.infer(X))
        return np.array(Y_batch).squeeze()

    def backprop(self, Y_target, signals):
        """Back-propagate the loss to calculate gradients for each weight.

        Returns a list of gradients, one for each layer.

        """
        grads = []
        Y_pred = signals[-1][-1]

        # for s in signals:
        #    print(s.shape)

        # Gradient of loss w.r.t network output.
        G = self.loss_grad_fn(Y_target, Y_pred)

        i_linear = 1
        for lspec, (_, _, S) in (
                reversed(list(zip(self.layer_spec, signals[:-1])))):
            ltype = lspec["type"]
            if self.debug:
                print(f"Layer: {ltype}, G-shape: {G.shape}, S-shape: {S.shape}")

            if ltype == "relu":
                G = grad_relu(S) * G

            elif ltype == 'sigmoid':
                G = grad_sigmoid(S) * G

            elif ltype == 'linear':
                Wb = self.weights[-1 * i_linear]
                if self.debug:
                    print(f"Wb-shape: {Wb.shape}")

                # We know that formula for gradient of linear layer weights is
                # `dz/dw = x.G` and for bias it is `dz/db = x`. We can do an
                # outer product of input vector (augmented with 1.0 for bias),
                # with the G to produce gradient of each weight and bias of the
                # layer.
                S_ = np.append(S, [[1.0]], axis=0).reshape(-1, 1)
                grad = np.einsum('ik,jl->ij', G, S_)
                grads.append(grad)

                # Slide G matrix of shape (n_outp, 1) through the columns of
                # weight matrix of shape (n_outp, n_inp) and do a sum after
                # element-wise multiplication. This should produce the G for
                # new-iteration with a shape of (n_inp, 1).
                G = np.einsum('ki,kj->ij', Wb[:, :-1], G)

                i_linear += 1

            else:
                raise ValueError("Invalid layer type during back-prop.")

        # print(grads)
        return list(reversed(grads))

    def update_weights(self, grads, lr=0.1):
        """Update weights using SGD."""
        for i, grad in enumerate(grads):
            delta = grad * lr
            self.weights[i] -= delta


def train_network(
        net,
        train_X, train_Y,
        calc_accuracy_fn,
        # Valid values are: stochastic, mini-batch, batch
        mode='stochastic',
        epochs=500,
        snapshot_callback=None,
        learning_rate=0.01,
        print_progress=True):

    if mode == 'stochastic':

        for n in range(1, epochs + 1):
            loss_hist = []
            for X, Y in zip(train_X, train_Y):
                # Do a forward pass.
                signals = net.forward_pass(X)

                # Save loss to history.
                loss = net.get_loss(Y, signals)
                loss_hist.append(loss)

                # Run backpropagation to generate gradients.
                grads = net.backprop(Y, signals)

                # Update weights using gradient descent.
                net.update_weights(grads, lr=learning_rate)

            # print(loss_hist)
            avg_loss = np.mean(np.array(loss_hist).squeeze(), axis=0)
            pred_Y = net.infer_batch(train_X)
            acc = calc_accuracy_fn(train_Y, pred_Y)

            if snapshot_callback:
                snapshot_callback(net, {"epoch": n})

            if print_progress:
                print(
                    "Epoch: {}, Avg. loss: {}, Accuracy: {}"
                    .format(n, avg_loss, acc))

    elif mode == 'batch':

        for n in range(1, epochs + 1):
            loss_hist = []
            grads_hist = []
            for X, Y in zip(train_X, train_Y):
                # Do a forward pass.
                signals = net.forward_pass(X)

                # Save loss to history.
                loss = net.get_loss(Y, signals)
                loss_hist.append(loss)

                # Run backpropagation to generate gradients.
                grads = net.backprop(Y, signals)
                grads_hist.append(grads)

            # Calculate mean gradient at each layer.
            avg_grads = []
            for i in range(len(grads_hist[0])):
                avg_grads.append(
                    np.mean([gh[i] for gh in grads_hist], axis=0))

            # Update weights using gradient descent.
            net.update_weights(avg_grads, lr=learning_rate)

            # print(loss_hist)
            avg_loss = np.mean(np.array(loss_hist).squeeze(), axis=0)
            pred_Y = net.infer_batch(train_X)
            acc = calc_accuracy_fn(train_Y, pred_Y)

            if snapshot_callback:
                snapshot_callback(net, {"epoch": n})

            if print_progress:
                print(
                    "Epoch: {}, Avg. loss: {}, Accuracy: {}"
                    .format(n, avg_loss, acc))

    else:
        raise ValueError("Unsupported training mode.")

    return net


def test_regression_sigmoid_1():
    """A test network with sigmoid activation function to do regression."""
    mlp = MLP(
        input_dim=1,
        layer_spec=[
            {"type": "linear", "n_neurons": 10},
            {"type": "sigmoid"},
            {"type": "linear", "n_neurons": 5},
            {"type": "sigmoid"},
            {"type": "linear", "n_neurons": 2},
        ],
        debug=False)

    # Create training data.
    train_X = np.linspace(0.0, 5.0, 10).reshape(-1, 1)
    train_Y1 = (10 + 3 * train_X) / (train_X + 0.5)
    train_Y2 = 2 * train_X + np.square(train_X)
    train_Y = np.hstack((train_Y1, train_Y2))

    train_network(
        mlp, train_X, train_Y, r2_score, epochs=500, learning_rate=0.01)


def test_regression_relu_1():
    """A test network with relu activation function to do regression."""
    mlp = MLP(
        input_dim=1,
        layer_spec=[
            {"type": "linear", "n_neurons": 5},
            {"type": "relu"},
            {"type": "linear", "n_neurons": 20},
            {"type": "relu"},
            {"type": "linear", "n_neurons": 2},
        ],
        debug=False)

    # Create training data.
    train_X = np.linspace(0.0, 5.0, 10).reshape(-1, 1)
    train_Y1 = (10 + 3 * train_X) / (train_X + 0.5)
    train_Y2 = 2 * train_X + np.square(train_X)
    train_Y = np.hstack((train_Y1, train_Y2))

    train_network(
        mlp, train_X, train_Y, r2_score, epochs=500, learning_rate=0.001)


if __name__ == "__main__":
    test_regression_sigmoid_1()
    # test_regression_relu_1()
