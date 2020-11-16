==============================================================================
NeuralFlow: modeling neural activity with continuous latent Langevin dynamics
==============================================================================

NeuralFlow is a Python package for modeling neural spiking activity with continuous latent Langevin dynamics. 
The driving force is optimized from data by gradient-descent algorithm directly in the space of continuous functions.
Modeling results can be interpreted within the dynamical system theory.

We also provide a model selection tool for choosing the correct model from a sequence produced by gradient descent.
Our method is based on direct feature comparison of models fitted on different data samples and aims for finding consistent
models with the correct interpretation.
In the associated article we demonstrate that our feature consistency analysis outperforms conventional 
validation-based model selection methods when the goal is finding the model with the correct interpretation.  
For mode information, see Genkin and Engel (2020) [#Genkin2020]_.


.. toctree::
   :maxdepth: 2
   

   installation
   examples
   implementation




Reference
----------

.. [#Genkin2020] `Genkin, M., Engel, T.A. Moving beyond generalization to accurate interpretation of flexible models. Nat Mach Intell 2, 674–683 (2020). <https://www.nature.com/articles/s42256-020-00242-6>`_
