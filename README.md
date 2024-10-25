#  NeuralFlow - version 3

## Short description

Computational framework for modeling neural activity with continuous latent Langevin dynamics. 

Quick installation: ```pip install git+https://github.com/engellab/neuralflow```

The source code for the following publications:

1) **M Genkin, KV Shenoy, C Chandrasekaran, TA Engel, The dynamics and geometry of choice in premotor cortex, bioRxiv (2023)** 

**Link:** https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10401920/

2) **Genkin, M., Hughes, O. and Engel, T.A. Learning non-stationary Langevin dynamics from stochastic observations of latent trajectories. Nat Commun 12, 5986 (2021)**.

**Link:** https://rdcu.be/czqGP

3) **Genkin, M., Engel, T.A. Moving beyond generalization to accurate interpretation of flexible models. Nat Mach Intell 2, 674–683 (2020)**.  

**Link:** https://www.nature.com/articles/s42256-020-00242-6/

**Free access:** https://rdcu.be/b9cW3

## Installation
Package only: pip install git+https://github.com/engellab/neuralflow

Package with examples: 

    git clone https://github.com/engellab/neuralflow
    cd neuralflow
    pip install .

If you have issues with Cython extension, and want to use precomplied .c instead, open setup.py and change line 7 to USE_CYTHON = 0

## GPU support

If your platform has CUDA-enabled GPU, install cupy package. Then you can use
GPU device for optimization.
Package passes unit tests with cupy-cuda12x==12.2.0

## documentation

https://neuralflow.readthedocs.io/

## Getting started

See examples 

## Deep dive

See tests
