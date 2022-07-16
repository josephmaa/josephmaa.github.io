---
layout: posts
title:  "Detecting Traffic Flow"
excerpt: "Learning about the architecture of model focused on a spatio-temporal graph convolutional neural network." 
categories: research
---

In this paper by Bing Yu, Haoteng Yin, and Zhanxin Zhu, researchers from the Beijin Institute of Big Data Research and Center for Data Science from Peking University try to use a spatial-temporal graph convolutional network to tackle time series prediction. 

The authors claim that impractical assumptions and simplifications degrade prediction accuracy of medium and long term (30 minutes into the future) traffic prediction. The problem is formally described as 

> Traffic forecast is a typical time-series prediction problem, i.e. predicting the most likely traffic measurements (e.g. speed or traffic flow) in the next H time steps given the previous M traffic observations as:

$$\hat{v_{t+1}}, ...., \hat{v_{t+H}} = \text{argmax}_{v_{t+1},...,v_{t+H}} \text{log} P(v_{t+1},...,v_{t+H}|v_{t-M+1},...,v_{t})$$
> where $$v_{t} \in \mathbb{R}^{n}$$ is an observation vector of n road segments at time step t, each element of which records historical observation for a single road segment.

Interestingly, due to the way they chose to structure the data, their weighted adjacency matrix $$W \in \mathbb{R}^{n x n}$$, which could require a large amount of memory.

#### Defining a convolution on a graph:

This paper re-introduced the idea of convolutions on graph. They use the standard definition of the convolution:

$$\Theta_{*G}x=\Theta(L)x=\Theta(U\Lambda U^{T})x=U\Theta(\Lambda)U^{T}x$$

where L is the normalized graph Laplacian is defined as $$L = I_{n} - D^{-\frac{1}{2}} W D^{-\frac{1}{2}} = U\Lambda U_{T} \in \mathbb{R}^{n x n}$$. The Laplacian is a symmetric matrix, so it is guaranted to be diagonalizable and $$\Lambda \in \mathbb{R}^{n x n}$$ is the diagnoal matrix of eigenvalues of L. The paper by Kipf and Welling introduces this operation in 2016. [^1]

#### Architecture:

The proposed architecture of the STGCN is comprised of two gated convolutional layer with one spatial graph convolutional layer nested between them as can be seen in the second block of the figure below.

<img src="{{site.baseurl}}/images/2022-07-13-detecting-traffic-flow/architecture.png" style="display: block; margin: 5% 0% 5% 0%;">

In Tensorflow, the implementation of the convolutional layers is pretty easy to use [^2] i.e.

```
def _build(self):

        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=True,
                                            logging=self.logging))
```
The authors claim that due to the graph Fourier basis, convolutions through matrix multiplications can be expensive since they're $$O(n^{2})$$, so they use two approximations to speed up calculations.

##### 1. Chebyshev Polynomial Approximations:



##### 2. 1st order Approximations

[^1]: https://tkipf.github.io/graph-convolutional-networks/
[^2]: https://github.com/tkipf/gcn/blob/master/gcn/models.py