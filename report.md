# CSCE 790 Neural Networks and Their Applications Homework 1 Report

### Author: `Titon Barua <baruat@email.sc.edu>**

### Answer 4

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






### Answer 7

To be linear, a function should satisfy this two properties. For any $a,b \in \mathbb{R}$ a function $f: \mathbb{R}^n \rightarrow \mathbb{R}^m$ is linear if and only if,

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
