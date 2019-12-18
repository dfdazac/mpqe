## Encoding logical queries with R-GCNs

This repository contains the implementation of Message Passing Query Embedding (MPQE), a method to answer complext queries over knowledge graphs using embeddings. 

<div align="center">
<img src='img/qrgcn.png'>
</div>

### Requirements

- PyTorch
- PyTorch Geometric
- Sacred

### Instructions

Clone the repository and run python `-m netquery.train` to train MPQE with a sum aggregation function.
