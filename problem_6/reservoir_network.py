"""This module was an attempt at implementing a reservoir network as described
in the paper -

'Reservoir observers: Model-free inference of unmeasured variables in chaotic systems, Zhixin Lu ; Jaideep Pathak ; Brian Hunt; Michelle Girvan; Roger Brockett; Edward Ott'

NOTE: The implementation does not work!!! There is a serious bug somewhere in
the implementation, preventing training procedure to be successful.

"""

import math
import torch
import numpy as np


def create_input_weight_matrix(n, m, sigma):
    """Initialize input weight matrix using the strategy described in paper.

    Args:
    - n: Number of reservoir nodes.
    - m: Number of scalar inputs.
    - sigma: Scale of the input weights.

    """
    W_in = torch.zeros((n, m), dtype=torch.double)
    # Initialize W_in in a way so that each row will just have a single
    # non-zero element. The non-zero element will is uniformly drawn between
    # -sigma to sigma. This ensures that each reservoir node is connected to a
    # single input signal.
    n, m = W_in.shape
    rand_col_indices = torch.randint(low=0, high=m, size=(n,))
    rand_input_weights = sigma * (
        (torch.rand(size=(n,), dtype=torch.double) - 0.5) / 0.5)
    W_in[range(n), rand_col_indices] = rand_input_weights
    return W_in


def create_adjacency_matrix(n, m, d, rho):
    """Create an adjacency matrix using the strategy described in paper.

    Args:
    - n: Number of reservoir nodes.
    - m: Number of scalar inputs.
    - d: Average degree of a reservoir node.
    - rho: Spectral radius.
    """
    # Initialize the adjacency matrix, A using Erdos-Renyi model.
    A = torch.zeros((n, n), dtype=torch.double)
    p = d / n  # Probability of an edge being connected.
    n_possible_edges = (n * n)
    dice_roll = torch.rand(size=(n_possible_edges,))
    # print(dice_roll.shape)
    k = 0
    for i in range(n):
        for j in range(i, n):
            if dice_roll[k] < p:
                # Choose a uniform value between -1 and 1.
                v = (torch.rand((1,)).item() - 0.5) / 0.5
                A[i, j] = A[j, i] = v
            k += 1

    # Calculate largest eigenvalue of A and find a scale factor to adjust
    # spectral radius of A.
    eighvals, _ = torch.linalg.eigh(A)
    largest_eigen_val = torch.abs(torch.max(eighvals.ravel()))
    scale_factor = rho / largest_eigen_val
    A = A * scale_factor

    return A



class ReservoirSystem(torch.nn.Module):

    def __init__(
            self,
            n,
            m,
            o,
            d,
            alpha,
            rho,
            sigma,
            eta):
        """Randomly initialize input layer and the reservoir nodes of a
        reservoir network.

        Args:
        - n: Number of reservoir nodes.
        - m: Number of input scalars.
        - o: Number of output scalars.
        - d: Average degree of a reservoir node.
        - alpha: Leakage rate.
        - rho: Spectral radius.
        - sigma: Scale of input weights.
        - eta: Input bias.

        """
        super().__init__()
        self.register_buffer("n", torch.tensor(n))
        self.register_buffer("m", torch.tensor(m))
        self.register_buffer("o", torch.tensor(o))
        self.register_buffer("d", torch.tensor(d))
        self.register_buffer("alpha", torch.tensor(alpha))
        self.register_buffer("rho", torch.tensor(rho))
        self.register_buffer("sigma", torch.tensor(sigma))
        self.register_buffer("eta", torch.tensor(eta))

        # Initialize input weights.
        W_in = create_input_weight_matrix(n, m, self.sigma)
        self.register_buffer("W_in", W_in)

        # Initialize adjacency matrix.
        A = create_adjacency_matrix(self.n, self.m, self.d, self.rho)
        self.register_buffer("A", A)

        # Initialize reservoir states.
        r_init = torch.zeros((n, 1), dtype=torch.double)
        torch.nn.init.normal_(r_init, mean=0.0, std=1.0)
        self.register_buffer("r_init", r_init)
        self.register_buffer("r", r_init)

    def forward(self, x):
        assert x.shape == (self.m, 1)
        assert self.r.shape == (self.n, 1)
        r_next = (
            (1.0 - self.alpha) * self.r
            + self.alpha * torch.tanh(
                self.W_in @ x
                + self.A @ self.r
                + self.eta))

        self.r = r_next
        return r_next

    def reset(self):
        self.r = self.r_init

    def reinitialize_reservoir_states(self):
        torch.nn.init.normal_(self.r_init, mean=0.0, std=1.0)
        self.reset()


class ReservoirNetwork(torch.nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        o = kwargs["o"]  # Number of output scalars.
        n = kwargs["n"]  # Number of reservoir nodes.

        # Initialize output weight.
        W_out = torch.zeros((o, n), dtype=torch.double)
        torch.nn.init.uniform_(W_out, -1.0, 1.0)
        self.register_parameter("W_out", torch.nn.Parameter(W_out))

        # Initialize output bias vector.
        c = torch.zeros((o, 1), dtype=torch.double)
        torch.nn.init.uniform_(c, -1, 1.0)
        self.register_parameter("c", torch.nn.Parameter(c))

        self.reservoir_system = ReservoirSystem(**kwargs)

    def forward(self, x):
        r = self.reservoir_system(x)
        return self.W_out @ r + self.c


class CustomLoss(torch.nn.Module):

    def __init__(self, reservoir_network, del_t, beta):
        super().__init__()
        self.rsvnet = reservoir_network
        self.del_t = del_t
        self.beta = beta

        self.k = 0
        self.output_history = []

    def forward(self, predicted, expected):
        self.k += 1
        self.output_history.append((
            self.rsvnet.reservoir_system.r, torch.tensor(expected)))

        W_out = self.rsvnet.W_out

        # Calculate loss from historic reservoir states and outputs.
        loss = 0
        for r, s in self.output_history:
            v = W_out @ r + self.rsvnet.c - s
            loss += v.T @ v
        # loss /= self.k

        # Add regularization term.
        loss += (self.beta * torch.trace(W_out @ W_out.T))

        return loss.squeeze()

    def reset(self):
        self.output_history = []


model = ReservoirNetwork(
    n=400,
    m=1,
    o=2,
    d=20,
    sigma=1.0,
    rho=1.0,
    eta=1.0,
    alpha=0.2)
# print(list(model.parameters()))

criterion = CustomLoss(model, del_t=0.1, beta=1e-4)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)

data = np.loadtxt("data/rossler_norm.nptxt")[:, 1:]
train_data, test_data = data[:100], data[100:]

loss = math.inf

epoch = 0
while (loss > 1e-3):
    try:
        epoch += 1

        optimizer.zero_grad()
        model.reservoir_system.reset()
        criterion.reset()

        for i in range(len(train_data)):
            x = train_data[i, 0].reshape(-1, 1)
            s_exp = train_data[i, 1:].reshape((-1, 1))
            s_pred = model(x)

            loss = criterion(s_pred, s_exp)
            loss.backward()
            optimizer.step()
        print("Epoch: {}, Loss: {}".format(epoch, loss))
    except KeyboardInterrupt:
        break

# model.reservoir_system.reset()
# x_test = test_data[0, 0]
# s_test = test_data[0, 1:]
# s_outp = model(torch.tensor(x_test.reshape(1, 1)))
# print(s_test)
# print(s_outp)
