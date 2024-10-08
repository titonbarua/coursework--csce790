# CSCE 790 Neural Networks and Their Applications Homework 1 Report

Author: Titon Barua <baruat@email.sc.edu>


## Probem 1: MNIST Classifier

The code is available [here](https://github.com/titonbarua/coursework--csce790/blob/main/homework_1/problem_1/mnist_classifier.py). On top of the code described in the blogpost, I added some basic evaluation metrics. I also trained a second network with double the neurons for each layer. Both of the networks were trained for 15 epochs.

With the second network, there was a slight improvement (0.42%) in overall classification accuracy. But there were some drops in precision and recall for some of the classes. With just 15 epochs of training, it is hard to assign any meaning to this fluctuations in precision and recall.


### Network with \(128, 64\) neurons in hidden layers

- Average validation accuracy is 97.27%

Table: Precision and recall by class for first network

|Digit|0|1|2|3|4|5|6|7|8|9|
|:-----|-|-|-|-|-|-|-|-|-|-|
|Precision %| 97.8|98.4|98.9|95.0|97.1|98.8|97.9|96.6|96.3|96.1|
|Recall %| 98.6|99.3|96.0|98.3|97.7|93.3|97.5|97.4|96.8|97.2|


![Confusion matrix for network with (128, 64) neurons](./problem_1/confusion_matrix_128x64.png)

### Network with \(256, 128\) neurons in hidden layers

Table: Precision and recall by class for second network

|Digit|0|1|2|3|4|5|6|7|8|9|
|:-----|-|-|-|-|-|-|-|-|-|-|
|Precision %|98.4|99.3|99.0|94.0|98.5|99.1|97.6|96.6|97.1|97.7|
|Recall %|98.7|98.7|97.0|98.9|97.6|95.4|98.1|98.3|97.2|96.6|

![Confusion matrix for network with (256, 128) neurons](./problem_1/confusion_matrix_256x128.png)

### Code Flow

- Download MNIST dataset from internet.
- Normalize the data and split them into batches.
- Build a fully connected neural network with this layering:
  - A fully connected layer
  - ReLU activation
  - Another fully connected layer
  - ReLU activation
  - Last fully connected layer
  - Softmax function convert the network output as probablities.
- Train the model using training data and save the model in the disk.
- Predict validation set and convert prediction from probabilities
  to class label by taking the argument of maximum probability.
- Evaluate performance using utilities from `scikit-learn.metrics` library.

<!--
Network: 128x64
Average accuracy: 0.9727
Precision:
	Class '0': 0.978
	Class '1': 0.984
	Class '2': 0.989
	Class '3': 0.950
	Class '4': 0.971
	Class '5': 0.988
	Class '6': 0.979
	Class '7': 0.966
	Class '8': 0.963
	Class '9': 0.961
Recall:
	Class '0': 0.986
	Class '1': 0.993
	Class '2': 0.960
	Class '3': 0.983
	Class '4': 0.977
	Class '5': 0.933
	Class '6': 0.975
	Class '7': 0.974
	Class '8': 0.968
	Class '9': 0.972


Network: 256x128
Average accuracy: 0.9769
Precision:
	Class '0': 0.984
	Class '1': 0.993
	Class '2': 0.990
	Class '3': 0.940
	Class '4': 0.985
	Class '5': 0.991
	Class '6': 0.976
	Class '7': 0.966
	Class '8': 0.971
	Class '9': 0.977
Recall:
	Class '0': 0.987
	Class '1': 0.987
	Class '2': 0.970
	Class '3': 0.989
	Class '4': 0.976
	Class '5': 0.954
	Class '6': 0.981
	Class '7': 0.983
	Class '8': 0.972
	Class '9': 0.966
-->


### Problem 2: Perceptron Response Graphs

Graph generating code is available [here](https://github.com/titonbarua/coursework--csce790/blob/main/homework_1/problem_2/plot.py).

![](./problem_2/perceptron_graph_a.pdf)

![](./problem_2/perceptron_graph_b.pdf)

\pagebreak

### Problem 3: CNN from Scratch

The code is available [here](https://github.com/titonbarua/coursework--csce790/blob/main/homework_1/problem_3/nn_from_scratch.py). I did not like some of the implementations of the author and took the liberty to write them in my own way.

- `relu`: Implemented by searching and zero-ing negative values with `numpy.argwhere`.
- `conv`: Used `scipy.signal.convolve2d` as the baseline convolution implementation.
- `max_pooling`: Used `block_reduce` function from `skimage.measure`.

Note: Colorization of the grayscale feature maps is done by applying `viridis` colormap.

![Layer1](./problem_3/layer1.pdf){width=50%}

![Layer2](./problem_3/layer2.pdf){width=80%}

![Layer3](./problem_3/layer3.pdf){width=80%}

\pagebreak

#### Algorithm

1. Take a color image and convert it to grayscale.
2. Create and apply two fixed convolutional filters, one for vertical edges; another for horizontal edges.
3. Apply `ReLU` activation function on feature maps from step 2.
4. Apply max-pooling to reduce size of feature maps from step 3.

For second and third layer, filters in step-2 were replaced by random convolution kernels.


### Problem 4: Learning Techniques

**Hebbian learning** is a biologically inspired learning rule. The core concept is that, given some randomly initialized neurons, if some training data activates two connected nodes simultaneously, their connecting weights should be enforced or increased. This iterative strengthening eventually leads to strong clustering of the hidden and output neurons.

**Perceptron update rule** is a method of training a single perceptron. The
  perceptron starts with zero or randomly initialized weights and bias. For each
  training sample, the difference between desired output and the current output
  is calculated. Until convergence, weights are updated in proportion to both
  the error and the input feature. The bias is updated only in proportion to the
  error. This training method fails if the data is not linearly separable.

**Delta learning rule** is a method of training a perceptron. The weights are
  iteratively updated in proportion to the negative of the gradient of the least
  square error of the perceptron output. In case of a non-linear data-set, delta
  learning produces a linear separation that minimizes the least square error.
  Delta learning is the precursor to gradient descent which is the most
  practical neural network training approach.

**Correlation learning rule** is a modification of hebbian learning. In contrast
  to hebbian learning, which clusters the neurons in an unsupervised manner,
  correlation learning updates the output neurons to produce desired outputs in
  a supervised setting.

**Out Star learning rule** is a learning rule for layers of neurons. To my
understanding, weights are updated in a bottom-up fashion where output neurons'
drive the weights of the synaptic connections. The article does not go into
mechanistic details of the procedure and the resources on the web related this
method is rare.

**Competitive learning rule** is an approach where instead of updating the
weights, the learning happens by selecting output neurons with strongest
responses. The network is initialized with random weights and biases. For a
class of inputs, the output neuron with the maximal response is declared the
winner and assigned to that particular class. The weights of the specific neuron
is then updated to further strengthen it's response that class.

\pagebreak

### Problem 5: Graph Neural Networks

This tutorial was a fire-hose of graph neural network based techniques. The
author introduced both the graph-level and node-level GNN techniques and
compared them to a fully connected network. My implementation (a.k.a copy-paste)
of the code is available
[here](https://github.com/titonbarua/coursework--csce790/blob/main/homework_1/problem_5/graph_neural_networks.py).
I used my laptop with an NVIDIA GPU for training.

On the `Cora` dataset, the fully connected network performed `60.60%` on test accuracy while demo node-level GNN model performed `81.10%`.

```python
train_and_test_mlp()
# Train accuracy: 97.14%
# Val accuracy:   50.40%
# Test accuracy:  60.60%

train_and_test_gnn()
# Train accuracy: 100.00%
# Val accuracy:   76.20%
# Test accuracy:  81.10%
```

On `mutag` dataset, the graph-level GNN performed `92.11%` test accuracy.

```python
# Train performance: 91.29%
# Test performance:  92.11%
```

### General Ideas of GNN

- A graph works as the input. Each node has some feature vector attached. The
  relationship between the nodes is expressed using an adjacency matrix. The
  matrix is modified to have self-connection.

- To normalize importance of well connected nodes, the matrix is multiplied on
  both sides by a diagonal matrix. The structure seems surprisingly similar to
  *graph laplacian* used in eigenvalue analysis based spectral clustering
  techniques.

- In each update step, a node receives messages from all the nodes adjacent to
  it. This is accomplished by multiplying the node features by a weight vector
  and the adjacency matrix. Optionally, an attention layer is used.

\pagebreak

### Problem 6: Reservoir Networks


#### Signal Generation
My implementation of the ODE solver for the chaotic systems is available [here](https://github.com/titonbarua/coursework--csce790/blob/main/homework_1/problem_6/data_gen.py). I used different initial conditions compared to the paper, resulting in a different solution. Each time step was solved using 1000 iterations of first-order euler approximation.

For the lorenz system, I was struggling initially as the system was seemingly
diverging too quickly beyond double precision float capabilities. This lead me
down the rabbit hole. Eventually I discovered the issue. Third equation of the
lorenz system should be $$\frac{dz}{dt} = -\frac{8}{3}z + xy$$. A minus sign is
missing in the paper. This is confirmed by the code associated with the paper
where the mistake was corrected.

![Simulation of the Rossler System](./problem_6/graphs/rossler_norm.pdf){width=80%}

![Simulation of the Lorenz System](./problem_6/graphs/lorenz_norm.pdf){width=80%}


#### Neural Network

Misunderstanding the instructions of the assignment, I attempted to write my own
implementation of a reservoir network in pytorch. It was a really good
educational experience, as I had to implement the procedures for randomizing
(using Erdos–Renyi model) and normalizing (by spectral radius adjustment using
eigenvector analysis) the adjacency matrix.

Unfortunately, my implementation is not fully functional and it is not trainable
using SGD or Adam optimizers, as the training error does not go down as
expected. In the implementation of the paper, the authors used an analytical
solution for training. It's either the training procedure or I have made a
serious mistake in the network layout. My implementation is available
[here](https://github.com/titonbarua/coursework--csce790/blob/main/homework_1/problem_6/reservoir_network.py).

Sample training session:
```
Epoch: 1, Loss: 5950.533141365897
Epoch: 2, Loss: 1059.6563867851619
Epoch: 3, Loss: 250.11835225978672
Epoch: 4, Loss: 111.32755746470802
Epoch: 5, Loss: 87.94520995608738
Epoch: 6, Loss: 72.25983108251253
Epoch: 7, Loss: 66.19928406275031
Epoch: 8, Loss: 55.82608273224931
Epoch: 9, Loss: 50.86386251086432
Epoch: 10, Loss: 43.485477171082835
Epoch: 11, Loss: 39.59159264126123
Epoch: 12, Loss: 34.37489477303409
Epoch: 13, Loss: 31.346768354827603
Epoch: 14, Loss: 27.604957851696412
Epoch: 15, Loss: 25.24027291437891
Epoch: 16, Loss: 22.511169715716854
Epoch: 17, Loss: 20.65837191540538
Epoch: 18, Loss: 18.638831422310147
Epoch: 19, Loss: 17.185097555595775
Epoch: 20, Loss: 15.672879143036097
Epoch: 21, Loss: 14.532055982271547
Epoch: 22, Loss: 13.388974709606227
Epoch: 23, Loss: 12.49405068095448
Epoch: 24, Loss: 11.623496979916386
Epoch: 25, Loss: 10.921855091522357
Epoch: 26, Loss: 10.254897512137008
Epoch: 27, Loss: 9.705070717791097
Epoch: 28, Loss: 9.191659774708898
Epoch: 29, Loss: 8.760947519684107
Epoch: 30, Loss: 8.364206479398382
Epoch: 31, Loss: 8.02684182505463
Epoch: 32, Loss: 7.719268806313931
Epoch: 33, Loss: 7.454975469212729
Epoch: 34, Loss: 7.215857388623752
Epoch: 35, Loss: 7.0087028585543365
Epoch: 36, Loss: 6.822316107624026
Epoch: 37, Loss: 6.659798129349782
```


\pagebreak


### Problem 7: Linearity Test

A function $f: \mathbb{R}^n \rightarrow \mathbb{R}^m$ is linear if and only if for all $a \in \mathbb{R}$,

$$
\begin{split}
f(ax) &= a f(x)\\
f(x + y) &= f(x) + f(y)
\end{split}
$$

**a)** For $p \in \mathbb{R}^+$,

$$
\begin{split}
f(-px) &= |-px|\\
       &= |-p||x|\\
       &= p|x|\\
-p f(x) &= -p |x|
\end{split}
$$

Since $-p|x| \neq p|x|$, this function $f(x) = |x|$ is not linear.


**b)** For $x, y \in \mathbb{R}$,

$$
\begin{split}
f(x + y) &= (x + y)^2 + 2.(x + y) + 2\\
         &= x^2 + 2.xy + y^2 + 2x + 2y + 2\\
f(x) + f(y) &= x^2 + 4x + 2 + y^2 + 4y + 2\\
            &= x^2 + y^2 + 4x + 4y + 4
\end{split}
$$

Clearly, $f(x + y) \neq f(x) + f(y)$. Hence, $f(x) = x^2 + 2x + 2$ is not linear.


**c)** For $x, y \in \mathbb{R}; x \neq 0 ; y \neq 0$,

$$
\begin{split}
f(x + y) &= \frac{1}{x + y}\\
f(x) + f(y) &= \frac{1}{x} + \frac{1}{y}\\
            &= \frac{y + x}{xy}\\
\end{split}
$$

Since $\frac{1}{x + y} \neq \frac{y + x}{xy}$ in general, we can conclude that $f(x) = \frac{1}{x}$ is not linear.
