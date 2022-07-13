---
layout: post
title:  "Rumor Source Detection"
date:   2022-07-12 23:40:13 -0700
excerpt: "Comparing the pros and cons of different models evaluated by various centrality measures and evaluation metrics." 
categories: research
---

Epidemic model vs. Independent cascade model vs. Linear Threshold model

Centrality measures:
1. Degree centrality (indegrees)
2. Closeness centrality: mean shortest distance between a node and other nodes
3. Betweenness centrality: nodes appear in shortest paths
4. Jordan centrality: node with smallest max distance to contaminated and recovered nodes
5. Eigenvector centrality: Sume of degree centrality of neighbor nodes. Eigenvector of adjacency matrix coupled with largest eigenvalue

Evaluation metrics:
1. Execution time
2. F-measure and precision? (How are these measured based on accuracy of estimated sources?)
3. Distance error
4. Rank 

