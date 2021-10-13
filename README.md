#  NeuralFlow

## Short description

Computational framework for modeling neural activity with continuous latent Langevin dynamics. 

Quick installation: ```pip install git+https://github.com/engellab/neuralflow```

The source code for the following publications:

1) **Genkin, M., Hughes, O. and Engel, T.A., 2020. Learning non-stationary Langevin dynamics from stochastic observations of latent trajectories. Nat Commun 12, 5986 (2021)**.

**Link:** https://rdcu.be/czqGP

2) **Genkin, M., Engel, T.A. Moving beyond generalization to accurate interpretation of flexible models. Nat Mach Intell 2, 674–683 (2020)**.  

**Link:** https://www.nature.com/articles/s42256-020-00242-6/

**Free access:** https://rdcu.be/b9cW3

## Installation and documentation

https://neuralflow.readthedocs.io/

## Tutorial

### Part 1: Data format

Convert data from the spike times format to the ISI format.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/engellab/neuralflow/blob/master/tutorials/CCN2021/Exercises/Ex1_Data_Format.ipynb)

### Part 2: EnergyModel Class

Create EnergyModel class and visualize the framework parameters.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/engellab/neuralflow/blob/master/tutorials/CCN2021/Exercises/Ex2_EnergyModel_class.ipynb)

### Part 3: Synthetic data generation 

Generate synthetic data and latent trajectories from the ramping dynamics and visualize the latent trajectories, firing rate along these trajectories, and the spike rasters.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/engellab/neuralflow/blob/master/tutorials/CCN2021/Exercises/Ex3_Data_Generation.ipynb)

### Part 4: Model Inference

Optimize a model potential on spike data generated from the ramping dynamics.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/engellab/neuralflow/blob/master/tutorials/CCN2021/Exercises/Ex4_Model_Optimization.ipynb)

### Part 5: Feature consistency analysis for model selection

Implement feature consistency analysis for model selection.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/engellab/neuralflow/blob/master/tutorials/CCN2021/Exercises/Ex5_Feature_Consistency_Analysis.ipynb)


