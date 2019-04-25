## Encoding logical queries with R-GCNs

This repository contains code for the project of the seminar in *Combining Symbolic and Statistical Methods in AI* at the UvA.

In [Hamilton et al., Embedding Logical Queries on Knowledge Graphs](https://arxiv.org/abs/1806.01445), the authors present a method to solve complex queries on knowledge graphs. The query is answered by encoding it with a sequential application of operators that follow the subgraph represented by the query. The encoded query, a continuous vector or *embedding*, is then used to solve the query by finding the nearest neighbor embedding over all entities in the knowledge graph. The code for this project is based on the [original implementation](https://github.com/williamleif/graphqembed).

My proposal consists of replacing the sequential encoding of the query with a single operator that acts on the query subgraph: a [Relational Graph Convolutional Network](https://arxiv.org/abs/1703.06103).



