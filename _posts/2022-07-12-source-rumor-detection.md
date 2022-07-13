---
layout: posts
title:  "Source detection of rumors in social network"
excerpt: "Comparing the pros and cons of different models evaluated by various centrality measures and evaluation metrics." 
---
I started reading a review recently about source detection of rumors in a social network to start to get a better idea of a project that I'm planning on working on in the near future. 

#### Models:

There were mainly three models mentioned
1. Epidemic model:\
(S: susceptible, I: infected, R: recovered)
- SI 
- SIS
- SIR
- SIRS
At every time point, infected nodes have a chance to infect their susceptible neighbors.
2. Independent cascade model:\
Main difference is that it only has one chance to activate inactive neighboring nodes. 
3. Linear thresold model:\
At each time-step inactive nodes are weighted based on aggregation of incoming neighbors. The node is activated upon reaching a threshold value.


#### Centrality measures:
1\. Degree centrality:\
Also known as the indegree of a node.\

2\. Closeness centrality:\
Defined as the average shortest distance between a node and other nodes. This was introduced in a classic paper by Linton Freeman [^1] and also referred to as the "independence" of a point. Linton mentions that a previous version of centrality measured by Sabidussi (1966) is actually inversely related to centrality. 

> $$d(p_{i}, p_{k}) = \text{the number of edges in the geodesic linking }p_{i}\text{ and }p_{k}$$

Sabidussi's measure of the decentrality of a point $$p_{k}$$ is:
> $$C_{c}(p_{k})^{-1} = \sum_{i=1}^{n}d(p_{i}, p_{k})$$

3\. Betweenness centrality:\
Nodes that appear in the shortest paths. The cited papers are fairly recent (2002, 2011, 2014)\

4\. Jordan centrality:\
Also known as the center of a graph. This is related to the idea of closeness centrality using the Sabidussi measure. Node with smallest max distance to contaminated and recovered nodes. The number of Jordan centers is equivalent to the radius of a graph. [^2]
<img src="{{site.url}}/images/2022-07-12-source-rumor-detection/graph_centers.svg" style="display: block; margin: 5% 0% 5% 0%;"/>

5\. Eigenvector centrality:\
Sum of degree centrality of neighbor nodes. It's worth remembering that since the adjacency matrix of an undirected simple graph is symmetric, it has a complete set of real eigenvalues and an orthogonal eigenvector basis. We take the largest eigenvalue here which is bounded above by the maximum degree. [^3]

$$\lambda_{1}v_{x} = (Av)_{x} = \sum_{y=1}^{n}A_{x,y}v_{y} \leq \sum_{y=1}^{n}A_{x,y}v_{x} = v_{x}deg(x)$$

<div align="center">
A short proof of the largest eigenvalue being bounded above.
</div>

#### Evaluation metrics:
1. Execution time:
2. Accuracy:
- Precision:
$$\frac{\text{true positives}}{\text{true positives} + \text{false positives}}$$
- Recall:
$$\frac{\text{true positives}}{\text{true positives} + \text{false negatives}}$$
- F-measure:
$$\frac{2*\text{precision}*\text{recall}}{\text{precision}+\text{recall}}$$\
The F-measure is the harmonic mean of precision and recall (rates). 
3. Distance error:
This is the shortest distance between an accurate source and estimated source found by an algorithm. If there are multiple competing vertices as the candidate for the source, it is computed as the mean shortest path distance between the real source and the top scorers.
4. Rank:\
This is a conservative ranking based on the most likely node to be the source. If there are any ties, the real node is considered to be below the false positives.[^4]
5. Gini coefficient:\
Measures the inequality of message distribution within the network.

#### Source identification:
1\. Network partitioning:
- Phase 1: Voronoi partitioning method
- Phase 2: Adjust two source estimator between adjacent partitions.
<img src="{{site.url}}/images/2022-07-12-source-rumor-detection/k-centers.png" style="display: block; margin: 5% 0% 5% 0%;"/>

2\. Ranking based:\
This introduces an interesting idea of reverse flow. [^5] In the Independent cascade model, active nodes have probability $$p$$ to turn adjacent nodes into active nodes as well. Hence, a forward flow has probability $$1-p$$ to stop at a current node. If we run the flow backwards from every node in multiple simulations, we can find nodes that are hit frequently by reverse flow, which are more likely to be attackers.

$$P(stop(u)) = \prod_{v\in{N_{u}^{-}\bigcap I}}(1-p_{vu})$$

3\. Community based:\
This paper introduces the idea of partitioning a graph into communities via reverse propagation.[^6] The paper claims that the leading eigenvector method takes advantage of a benefit function called $$modularity$$.

$$q = (\text{number of edges within communitiyes}) - (\text{expected number of edges})$$

The paper suggests maximizing this function with indicate better divisions. It mentions exhaustive maximization over all possible divisions, but I wonder if we could just use sparse cut algorithms.

4\. Approximation based:
This paper introduces the idea of a set resolving set (SRS) which is the smallest subset such that can identify infection sources without knowledge of the number of source nodes. The algorithm is a polynomial time greedy algorithm.

#### Future research challenges:
1. Cycle detection in complex networks
2. Heterogenous diffusion
3. Multiple rumor sources
4. Real-time data collection and data snapshots
5. Dynamic temporal network evolution
6. Number of propagation sources

#### Conclusion:
I particularly was interested in the idea of the reverse flow and stochastic algorithms for approximating the source nodes for epidemic/rumor diffusion. It appears to be an open problem to determine the number of sources in any network. While some of the defining measures of centrality within a graph were defined in the 1950s, there are also a lot of recent methods addressing approximating sources with more compute power, especially in light of COVID tracking in the past two years.

[^1]: Freeman, Linton C. “Centrality in Social Networks Conceptual Clarification.” Social Networks 1, no. 3 (January 1, 1978): 215–39. https://doi.org/10.1016/0378-8733(78)90021-7.
[^2]: https://mathworld.wolfram.com/GraphCenter.html
[^3]: https://en.wikipedia.org/wiki/Adjacency_matrix
[^4]: Paluch, Robert, Xiaoyan Lu, Krzysztof Suchecki, Bolesław K. Szymański, and Janusz A. Hołyst. “Fast and Accurate Detection of Spread Source in Large Complex Networks.” Scientific Reports 8, no. 1 (February 6, 2018): 2508. https://doi.org/10.1038/s41598-018-20546-3.
[^5]: Nguyen, Dung T., Nam P. Nguyen, and My T. Thai. “Sources of Misinformation in Online Social Networks: Who to Suspect?” In MILCOM 2012 - 2012 IEEE Military Communications Conference, 1–6, 2012. https://doi.org/10.1109/MILCOM.2012.6415780.\
[^6]: Zang, Wenyu, Peng Zhang, Chuan Zhou, and Li Guo. “Discovering Multiple Diffusion Source Nodes in Social Networks.” Procedia Computer Science, 2014 International Conference on Computational Science, 29 (January 1, 2014): 443–52. https://doi.org/10.1016/j.procs.2014.05.040.



