# CSCE 790 Neural Networks and Their Applications Homework 1 Report

### Author: `Titon Barua <baruat@email.sc.edu>`


### Answer 6

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
