# causalkit: A Rust Package for Causal Inference

**causalkit** is a rust package that implements a set of algorithms for causal inference modeling. Currently, it supports two tree-based algorithms for individual treatment effect estimation with binary response and continuous response respectively. Given an intervention `T` and response `Y`, it estimates the individual treatment effect (ITE) for each sample. Both algorithms support multiple treatments, e.g. for an intervention with K treatments (control, treatment_1, treatment_2, ..., treatment_K), it will output the uplift in treatment effect for each treatment_i against control. 

* **Tree-based algorithms**
    * Binary Response: Uplift Random Forest on KL-Divergence [[1]](#Literature)
    * Continuous Response: Uplift Random Forest with honest estimation [[2]](#Literature)

This modeling approach can be applied to problems where the personalized effectiveness of an intervention `T` to target users is concerned.

# Installation

It requires python version `>=3.8`. To use it in python, install as follows
```
pip install causalkit
```

You can also download the source code from this github repository and build the python library by yourself via [maturin](https://github.com/PyO3/maturin).

# Usage

Please check the jupyter notebook in the example folder to learn how to use the library.

# References

## Literature
1. Stefan Wager, Susan Athey. "Estimation and Inference of Heterogeneous Treatment Effects using Random Forests" arXiv preprint arXiv:1510.04342 (2015).
2. Susan Athey, Guido Imbens. "Recursive Partitioning for Heterogeneous Causal Effects" arXiv preprint arXiv:1504.01132 (2015).
