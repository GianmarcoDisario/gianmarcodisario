---
layout: post
title: "Ignoring the Curse of Dimensionality"
date: 2024-02-12
categories: post
---

# Ignoring the Curse of Dimensionality

> **Note:** _This is a blog I have adjusted slightly from a blog I wrote back in 2019. So it could be that recent developments have shed some more light on the questions I outlined at the end. In case you know know how this field has evolved in the last few years, please reach out and let me know! :wink:_

While walking around on holiday in back in 2019 Mexico I was listening to [Lex Fridman's Artificial Intelligence](https://lexfridman.com/ai/) podcast (when it was still called the AI podcast) with the guest [Tomaso Poggio](https://en.wikipedia.org/wiki/Tomaso_Poggio) where he said something really interesting that struck a chord with me:

<center>
<iframe width="560" height="315" src="https://www.youtube.com/embed/aSyZvBrPAyk?si=tTwWBTDRl1NiWy6n&amp;start=2503&end=2559" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
</center>

<center>
<i>It has been one of the puzzles about neural networks. <br> How can you get something that really works when you have so much freedom?</i>
</center>
<br> 

The main question I was wondering is why can we train such big neural networks where the amount of parameters in our system is greater than the amount of training samples. Let us for example look at a classic neural network structure of [ResNet-152](https://arxiv.org/abs/1512.03385). ResNet-152 is a 152-layer Residual Neural Network with over 60.200.000 learnable parameters.

It is trained on the images of [ImageNet](http://www.image-net.org/). These images have a size of 224x224 pixels and channels for Red Green and Blue (RGB), which means that there are (224x224x3) 150.528 activations of pixels, resulting in an equivalent input size. ImageNet furthermore has 1.280.000 training images with 1.000 labels/classes. If the images are perfectly balanced between the different classes, then in the ideal case there are 1.280 images per class.

Everything that is explained about dimensionality in classical machine learning says that learning a model build up of 60 million learnable parameters with a 150k dimensional input and just 1.280 training samples per class should never be able to learn properly, let alone generalize well. 

In this blogpost I would like to dive into why is it that the really advanced neural networks work so well while they have vastly more parameters than training samples. To do this, we will need some background of the classical *Curse of Dimensionality*, as it is often taught in classical textbooks.

Subsequently, I will take a look at neural networks and explain the Universal Approximation Theorem. Since the Universal Approximation Theorem only holds for single layer neural networks, we will look if it possible to extend this unique property of single layer neural networks to deeper networks. Once we have these building blocks in place, we are going to combine them to see how they can be used to approximate the way giant neural networks can still learn.

# Curse of Dimensionality
As most data scientists are familiar with the main concepts of the curse of dimensionality, they will not be set out in this blogpost. In case you are not that familiar with it, the following video lecture covers it in the best 30 minute version that I have come across (From about 01:00 until 31:00).

<center>
<iframe width="560" height="315" src="https://www.youtube.com/embed/BbYV8UfMJSA?si=DxY1wVyrLNKXwg11&amp;start=60" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
</center>

<br>

Here I will briefly explain the part of the curse of dimensionality which is useful to us. A consequence of the curse of dimensionality is that in general you try to limit the amount of features you use to describe the relationship between input and output of the model. The curse of dimensionality mainly takes its toll on data with high multi-dimensional covariance. When making a machine learning model, it is assumed that there is sufficient structure (sufficiently low covariance/noise across different features) in the data to allow for a proper mapping of input to output by an approximate mathematical function. It is often the case that when a high-dimensional space is used (many features), that the useful relationship lives on a lower dimensional representation. This lower dimensional representation can either be a sub-space or a sub-manifold of the high-dimensional data (<a href='https://www.youtube.com/watch?v=BbYV8UfMJSA&t=665'>a more elaborate explanation of this idea can be seen here</a>). The reduction to this lower dimensional representation is achieved by a dimensionality reduction technique, or the machine learning model to be trained tries to implicitly learn this lower dimensional sub-space/-manifold.

# Why can we train a neural network with so many parameters

Since the intrinsic dimensionality of the data is lower than its representation through the features (even though it might need the high dimensionality for its sub-manifold representation), we realise that the Curse of Dimensionality does not necessarily have to pose a problem for most applications of data science. Now, we can take a look at neural networks.

## Properties of a single layer neural network

As in the case of almost any data science problem, we are trying to find the relationship between the input space, let us define it by $X \subseteq \mathbb{R}^{d}$, and the output space, defined by $Y \subseteq \mathbb{R}^{m}$. We will define this relationship as the non-linear function $f: X\rightarrow Y$. This means that for an input $x \in X$, the output can be calculated by $y = f(x)$ (where $y \in Y$). 

Let us not think of a (single layer) neural network in the classical sense of all the different nodes, but only as a functional mapping. This means that we are going to set up a function $F: X \rightarrow Y$ which is going to be the neural network. It will try to approach the nonlinear function $f(x)$. 

From the Universal Approximation Theorem [<a href='https://www.sciencedirect.com/science/article/abs/pii/089360809190009T'>1</a>], it is known that a 1-layer (or shallow) neural network can approximate any non-linear function arbitrarily well. The only thing that is never mentioned is how big actually the neural network has to be in order to approximate it well. Besides that, it is non-trivial that the Univeral Approximation Theorem extends to deeper (let alone convolutional, residual, etc.) networks as well.

## Degree of approximation

We will now approximate the size of a shallow neural network and call it the *degree of approximation*. Let $V_J$ be the set of all shallow networks with $J$ units(/neurons). It is assumed that networks of higher complexity include those of lower complexity, or $V_J \subseteq V_{J+1}$. The <i>degree of approximation</i> is then defined by

$$ 
\text{dist}(f,V_J) = \inf_{P\in V_J} ||f-P||. ~~~(1) 
$$

This definition is used to set up a metric by which we can measure how well a certain size of network is able to approximate a function. It is now provable that in the case it is desired that $\text{dist}(f,V_J) < \epsilon$ for any arbitrary $\epsilon$, the complexity of the network becomes in the order of $\mathcal{O}(\epsilon^{-\frac{1}{d}})$. Which means, if an accuracy is desired of within 10% of the original function, a network of size $10^d$ is needed to guarantee it [<a href='https://arxiv.org/abs/1611.00740'>2</a>], where $d$ was the dimensionality of the input. It should be noted that there are only $10^{86}$ atoms in the universe. Taking it back to the ImageNet case with 150.528 inputs, it does seem quite hard to find a feasible shallow network that can learn the 150.528 input to a single class output mapping of ImageNet. That's why we need more layers in the network.

## Extension to deep neural networks
It is familiar to everyone that most networks do have more layers than one, and that they provide in general better approximations of the function to be learned. What is not trivial however, is that the Universal Approximation Theorem also holds for deeper networks. From [<a href='https://arxiv.org/abs/1611.00740'>2</a>] it is also shown that the Universal Approximation Theorem can be extended to deeper networks, under two conditions.

### Condition 1. The function $f(x)$ is a compositional nature
The first condition is that $f(x)$ should be of a compositional nature. This  means you are able to write $f(x)$ as a function of functions, for example $f(x_1, x_2, x_3, x_4) = h_2 (h_{11}(x_1, x_2), h_{12}(x_3, x_4))$. This is quite a natural condition, because that is what the second layer of a neural network introduces on the first layer. The surprising thing is that there is no rigorous mathematical definition of when a function is decompositional, but it seems more of an issue on the crossroads of mathematics and philosophy [3-5]. There are hypotheses that everything related to the natural sciences has the property of functional decompositionality. There are also many qualitative arguments that language and images are (de)compositional in nature, which is quite natural to imagine. We can think of language as the building up from letter to word, to sentence, to paragraph, etc. Where each level might have different emergent properties of complex meaning. This is sometimes proposed why neural networks work so well in Natural Language Processing (NLP).

### Condition 2. The activation function $\sigma$ is smooth
The second condition is that the activation function $\sigma$ should be smooth (the derivative of a function should be continuous). The most used activation function nowadays is the ReLU function, whose derivative in $0$ is not well-defined. It would mean that this condition is not satisfied, but by invoking the <a href="https://en.wikipedia.org/wiki/Stone%E2%80%93Weierstrass_theorem">Weierstrass approximation theorem</a> on the ReLU function it can be uniformly approximated as closely as desired by a polynomial function (of high enough order). In figure 1, the ReLU activation function is approximated by a polynomial and the resulting network (trained on <a href="https://www.cs.toronto.edu/~kriz/cifar.html">CIFAR-10</a>) empirically has approximately the same performance [<a href="https://arxiv.org/abs/1703.09833">6</a>]. Hereby it is possible to say that the second condition is satisfied as well.

<center> <img src="/assets/images/2024-02-12-ignoring-cod/smoothedrelu.png"> <br> <i> figure 1. Approximating the ReLU activation function by a polynomial and the resulting performance of the network. [<a href="https://arxiv.org/abs/1703.09833">6</a>]</i></center> 

### Consequence
Now that we see that both conditions are met, it is possible to extend the Universal Approximation Theorem to deep networks as well. It also gives the possibility to compare the approximation power of shallow vs deep networks. When setting up a neural network, in general, we do not have the analytic form of function $f(x)$ we are trying to approximate, but just have the input and output data.

For demonstrational purposes, we now introduce the polynomial function $Q(x,y)$ as the function we are trying to approximate with the network. By doing this, we can compare shallow vs deep networks. For example, let us define $Q(x,y)$ as the following function

$$ 
Q(x,y) = \left(a x^2y^2 + b x^2y + c xy^2 + dx^2 + 2exy + fy^2 + 2gx + 2hy + i\right)^{2^{10}}.~~~~(2) 
$$

For a 1-layer network, 2049 neurons are needed to approximate $Q$ arbitrarily well, whereas an 11-layer network just needs 39 neurons (9-3-...-3) to approximate $Q$ arbitrarily well [7]. (The fact it can be approximated arbitrarily well is because a polynomial function without noise is used.)

## To Summarize
To this point, several important topics have been discussed. Firstly, we now know that the curse of dimensionality is not that big of a deal due to the relatively lower intrinsic dimensionality of the data.

The second point is that we now know through the univeral approximation theorem that shallow networks can learn any arbitrary function. Although to achieve an appropriate upper bound on the error of the approximation, a new type of <i>curse</i> came up; the shallow network had to become near infinitely wide for ImageNet type inputs. To circumvent this problem, we can prove that for smooth activation function (and the compositionality condition), the universal approximation theorem can be exteded to deep neural networks as well. The positive effect of making a network deeper is that it needs fewer neurons to perform as well as a (more) shallow network.

The final question that remains is: Why is it possible to find an optimum in a space with fewer samples than parameters to tune?

# Solutions of a deep neural network 
To address the final question, it is useful to know what constitutes a well performing neural network. In the ideal case, this is a neural network that has 0 (train and) test error. In the case of a neural network with an input size of $x \in \mathbb{R}^{N\times d}$, with $N$ samples and $d$-dimensional input. The output of size $y \in \mathbb{R}^{N\times m}$, the amount of weights $w \in \mathbb{R}^{K}$ and the depth of the network to be $L$. The performance of the network is in general approximated by a cost function,

$$ 
J(w, x) = \sum_{i=1}^{N}V\left(F(x_i, w), y_i  \right), ~~~~ (3) 
$$ 

where the optimum can be found if $\nabla_w J(w,x) = 0$ (which are $K$ equations). If a polynomial function is chosen as the loss function (for example the sum of squared errors $ J(w,x)=\sum_{i=1}^{N}(F(x_i, w)-y_i)^2 $), and the activation function for each neuron is the polynomial approximation of the ReLU (as used in fig. 1), by induction it can quite easily be shown that $\nabla_wJ(w,x)$ is a polynomial function as well. This gives us a set of polynomial equations of the form

$$ 
\left\{\begin{array}{rcl}
J'_{w_1} &=& \frac{\partial J}{\partial w_1} = 2 \sum_{i=1}^{N} (F(x_i,w)-y_i)\frac{\partial F(x_i,w)}{\partial w_1} =0, \\
J'_{w_2} &=& \frac{\partial J}{\partial w_2} = 2 \sum_{i=1}^{N} (F(x_i,w)-y_i)\frac{\partial F(x_i,w)}{\partial w_2} =0, \\
&\vdots& \\
J'_{w_K} &=& \frac{\partial J}{\partial w_K} = 2 \sum_{i=1}^{N} (F(x_i,w)-y_i)\frac{\partial F(x_i,w)}{\partial w_K} =0. \\
\end{array} \right. ~~~~~ (4) 
$$

Where the (global) optimum is found if $ J_{w_1}' = J_{w_2}' = \ldots = J_{w_K}' =0 $. Since these are all polynomial equations, we can make use of an old 18th century theorem from algebraic geometry, namely <a href="https://en.wikipedia.org/wiki/B%C3%A9zout%27s_theorem">Bézout's theorem</a>. It gives an approximation to the number of solutions for systems of polynomial equations. 

As noted earlier, an approximation is made of the ReLU function of polynomial degree $l$ (remember, $x^3 + 4x^2 - 2x$ is a polynomial of degree 3) with an error in approximation of $\delta$. It can be proven that there is an upper bound of $l^{NL}$ global optima with an estimate of $l^{\frac{NL}{2}}$ non-degenerate solutions to the set of equations (4) [<a href="https://arxiv.org/abs/1703.09833">6</a>, 8].

As it is known in practice that a neural network (almost) never completely achieves a global optimum, this might seem as a rather useless result. However, the high-dimensional landscape in the neighborhood of the global optima is scattered with well-performing local optima. These are referred to as basins of the landscape, because they propose separate basins of optimal behaviour [<a href="https://arxiv.org/abs/1703.09833">6</a>].

A different (quite elegant) way of interpreting the space is by imagining a fractal nature. This results in a so-called basin-fractal. Where there are still separate basins, but now follow a fractal pattern of decreasing local optima with so-called <i>walls</i> in between the different local optima and basins. Although there are indications of the landscape having a fractal nature, there are also contradicting empirical results. For now, this remains an hypothesis and for the technical discussion, the interested reader is referred to  [<a href="https://arxiv.org/abs/1703.09833">6</a>].


## Interpretive meaning
Ok, nice, all that abstract math with an estimation of global optima, but what does it mean in practice? It means that we can prove that by increasing the amount of parameters in our network (as well as the amount of training samples), we increase the amount of places in the high-dimensional space where absolute zeros exist of the loss function with specific settings of the weights. On the one hand, this seems intuitive; by introducing more and more parameters, the network has more possibilities of setting the weights in such a way that it optimizes the loss function. For example, for a <a href="https://www.cs.toronto.edu/~kriz/cifar.html">CIFAR-10</a> model, it is not unusual to obtain an upper bound approaching $2^{10^5}$ global optima [<a href="https://arxiv.org/abs/1703.09833">6</a>]. Sadly though, the achievability of a global optimum in polynomial time is still small [8]. But, as many global optima are surrounded by local optima, we are able to increase the amount of acceptable solutions for our neural network by increasing the amount of parameters.

This is an amazing result right?! By increasing the search space of the neural network, we are actually helping it to improve. We already knew it heuristically, but now we can actually explain why we are increasing the amount of parameters of a neural network. 

Some questions still remain though: How is it possible that, in such a high-dimensional space, our model doesn't heavily overfit and manages to generalize well? Of course, regularization techniques helps well to prevent overfitting, but nevertheless the amount of parameters remains amazingly high. 

Secondly, why is it able to learn as well as generalize so well, with a relatively trivial optimization algorithm like Stochastic Gradient Descent (SGD)? 

Quite some interesting recent papers have been written about these subjects [<a href="https://arxiv.org/abs/1903.04991">9</a>, <a href="https://arxiv.org/abs/1806.11379">10</a>, <a href="http://oastats.mit.edu/handle/1721.1/122014">11</a>], which might spark my interest for a future blog post...

# Bibliography
<i> Links are provided for papers which are easily freely accessible.</i>

<small>
 [<a href='https://www.sciencedirect.com/science/article/abs/pii/089360809190009T'>1</a>] Hornik, Kurt. "Approximation capabilities of multilayer feedforward networks." Neural networks 4.2 (1991): 251-257.<br>
[<a href='https://arxiv.org/abs/1611.00740'>2</a>] Poggio, Tomaso, et al. "Why and when can deep-but not shallow-networks avoid the curse of dimensionality: a review." International Journal of Automation and Computing 14.5 (2017): 503-519.<br>
[3] Koestler, Athur (1973), "The tree and the candle", in Gray, William; Rizzo, Nicholas D. (eds.), Unity Through Diversity: A Festschrift for Ludwig von Bertalanffy, New York: Gordon and Breach, pp. 287–314 <br>
[4] Simon, Herbert A. (1996), "The architecture of complexity: Hierarchic systems", The sciences of the artificial, Cambridge, Massachusetts: MIT Press, pp. 183–216. <br>
[5] Simon, Herbert A. (1963), "Causal Ordering and Identifiability", in Ando, Albert; Fisher, Franklin M.; Simon, Herbert A. (eds.), Essays on the Structure of Social Science Models, Cambridge, Massachusetts: MIT Press, pp. 5–31.<br>
[<a href="https://arxiv.org/abs/1703.09833">6</a>] Poggio, Tomaso, and Qianli Liao. Theory II: Landscape of the empirical risk in deep learning. Diss. Center for Brains, Minds and Machines (CBMM), arXiv, 2017. <br>
[7] H. N. Mhaskar, “Neural networks for optimal approximation of smooth
and analytic functions,” Neural Computation, vol. 8, no. 1, pp. 164–177,
1996 <br>
[8] M. Shub and S. Smale, “Complexity of bezout theorem v: Polynomial time,” Theoretical Computer Science,
no. 133, pp. 141–164, 1994. <br>
[<a href="https://arxiv.org/abs/1903.04991">9</a>] Banburski, Andrzej, et al. "Theory III: Dynamics and Generalization in Deep Networks." arXiv preprint arXiv:1903.04991 (2019). <br>
[<a href="https://arxiv.org/abs/1806.11379">10</a>] Poggio, Tomaso, et al. "Theory IIIb: Generalization in deep networks." arXiv preprint arXiv:1806.11379 (2018). <br>
[<a href="http://oastats.mit.edu/handle/1721.1/122014">11</a>] Poggio, Tomaso, Andrzej Banburski, and Qianli Liao. Theoretical Issues In Deep Networks. Center for Brains, Minds and Machines (CBMM), 2019.
</small>


