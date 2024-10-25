==============================================================================
NeuralFlow: modeling neural activity with continuous latent Langevin dynamics
==============================================================================

NeuralFlow is a Python package for modeling neural spiking activity with continuous latent Langevin dynamics. 
The driving force is optimized from data by gradient-descent algorithm directly in the space of continuous functions.
Non-stationary data distribution can be modelled by inferring initial distribution of the latent states and 
noise magnitude from data using absorbing or reflecting boundary conditions. 
Each neuron has its own tuning function f(x) that links neuron's firing rate to
the latent state.

The modeling results can be interpreted within the dynamical system theory.
The package includes optimization functions (ADAM and Gradient-descent), data generation functions for generating 
spike data from a specific Langevin model, and the core modules like spike data, PDE solver, and model. 
Viterbi algorithm for inferring the most probable path in also included.

In addition, we include feature consistency analysis for model selection based on feature consistency between
the models optimized on two data samples.

To get started, see examples. For deeper understanding of the code, refer to the unit tests. 
For mode information, see Genkin and Engel (2020) [#Genkin2020]_, Genkin, Hughes and Engel (2020) [#Genkin2021]_,
and Genkin et. al. (2023) [#Genkin2023]_

.. toctree::
   :maxdepth: 2
   

   installation
   examples
   implementation




References
----------

.. [#Genkin2020] `Genkin, M., Engel, T.A. Moving beyond generalization to accurate interpretation of flexible models. Nat Mach Intell 2, 674–683 (2020). <https://www.nature.com/articles/s42256-020-00242-6>`_

.. [#Genkin2021] `Genkin, M., Hughes, O. and Engel, T.A. Learning non-stationary Langevin dynamics from stochastic observations of latent trajectories. Nat Commun 12, 5986 (2021). <https://www.nature.com/articles/s41467-021-26202-1>`_

.. [#Genkin2023] `Genkin M, Shenoy KV, Chandrasekaran C, Engel TA. The dynamics and geometry of choice in premotor cortex. bioRxiv [Preprint] (2023) . <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10401920>`_