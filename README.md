# causalkit: A Rust Package for Causal Inference

**causalkit** is a rust package that implements a flexible random forest framework for causal modeling. Given an intervention `T` and response `Y`, it estimates the individual treatment effect (ITE) for each user. This modeling approach can be applied to problems where the personalized effectiveness of an intervention `T` to target users is concerned.

Currently, it supports two tree-based uplift algorithms for binary response and continuous response respectively. Both algorithms support multiple treatments, e.g. for an intervention with K treatments (control, treatment_1, treatment_2, ..., treatment_K), it will output the uplift for each treatment_i against control. it also provides a python package, so that the library could be used in python as well.

* **Tree-based algorithms**
    * Binary Response: Uplift Random Forest on KL-Divergence [[1]](#Literature)
    * Continuous Response: Uplift Random Forest with honest estimation [[2]](#Literature)

# Installation

## Rust

Download the source code from github [page]("https://github.com/rakutentech/causalkit.git") and build yourself.

## Python
It requires python version `>=3.8`. The library is available on PyPI. To use it in python, install as follows
```
pip install causalkit
```

# References

## Literature
1. Stefan Wager, Susan Athey. "Estimation and Inference of Heterogeneous Treatment Effects using Random Forests" arXiv preprint arXiv:1510.04342 (2015).
2. Susan Athey, Guido Imbens. "Recursive Partitioning for Heterogeneous Causal Effects" arXiv preprint arXiv:1504.01132 (2015).
