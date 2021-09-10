==============================================================================
NeuralFlow: modeling neural activity with continuous latent Langevin dynamics
==============================================================================

NeuralFlow is a Python package for modeling neural spiking activity with continuous latent Langevin dynamics. 
The driving force is optimized from data by gradient-descent algorithm directly in the space of continuous functions.
Non-stationary data distribution can be modelled by inferring initial distribution of the latent states and 
noise magnitude from data using absorbing or reflecting boundary conditions. The modeling results can be interpreted within the dynamical system theory.

We also a model selection tool for choosing the correct model from a sequence produced by gradient descent.
Our method is based on direct feature comparison of models fitted on different data samples and aims for finding consistent
models with the correct interpretation.
In the associated article we demonstrate that our feature consistency analysis outperforms conventional 
validation-based model selection methods when the goal is finding the model with the correct interpretation.  
For mode information, see Genkin and Engel (2020) [#Genkin2020]_, Genkin, Hughes and Engel (2020) [#Genkin2020Preprint]_.

.. toctree::
   :maxdepth: 2
   

   installation
   examples
   implementation




References
----------

.. [#Genkin2020] `Genkin, M., Engel, T.A. Moving beyond generalization to accurate interpretation of flexible models. Nat Mach Intell 2, 674–683 (2020). <https://www.nature.com/articles/s42256-020-00242-6>`_

.. [#Genkin2020Preprint] `Genkin, M., Hughes, O. and Engel, T.A. Learning non-stationary Langevin dynamics from stochastic observations of latent trajectories. arXiv preprint arXiv:2012.14944 (2020). <https://arxiv.org/abs/2012.14944>`_
