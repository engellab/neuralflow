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
For mode information, see Genkin and Engel (2019) [#Genkin2019]_.


.. toctree::
   :maxdepth: 2
   

   installation
   examples
   implementation




Reference
----------

.. [#Genkin2019] Mikhail Genkin, Tatiana A. Engel (2019) Beyond generalization: Enhancing accurate interpretation of flexible models. `BioRxiv (2019), p. 808261. <https://www.biorxiv.org/content/10.1101/808261v1>`_

