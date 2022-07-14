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



#### Architecture:

<img src="{{site.baseurl}}/images/2022-07-13-detecting-traffic-flow/architecture.png" style="display: block; margin: 5% 0% 5% 0%;">