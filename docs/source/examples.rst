.. _examples:

Examples
========

Here we provide examples from each of our papers  [#Genkin2020]_, [#Genkin2021]_, and [#Genkin2023]_. These examples can also be accessed with Jupiter notebook from our `GitHub repository <https://github.com/engellab/neuralflow/>`_ .

**************************************************************************
Moving beyond generalization to accurate interpretation of flexible models
**************************************************************************

The first example generates synthetic data from a double-well potential 
and uses this data to fit the model potential. It reproduces Figure 3 in the main text [#Genkin2020]_. 
The second example demonstrates our feature consistency method for model selection in the case of stationary dynamics. It reproduces Figure 5 in the main text [#Genkin2020]_.

.. toctree::
    :maxdepth: 2

    examples/2020_moving_beyond_generalization/Example1.ipynb
    examples/2020_moving_beyond_generalization/Example2.ipynb
    
*********************************************************************************************
Learning non-stationary Langevin dynamics from stochastic observations of latent trajectories
*********************************************************************************************

The first example generates synthetic data from the ramping dynamics, and optimizes the model potential on this data. Also the importance of various non-stationary components for accurate model inference is demonstrated. It reproduces Figures 2,3 in the main text [#Genkin2020preprint]_. 
The second example generates two synthetic datasets from ramping and stepping dynamics, and uses
this data to infer the model potentials. It also infers the model potential, the initial distribution of the latent states, and the noise magnitude from data generated from the ramping dynamics.  It reproduces Figure 4 in the main text [#Genkin2020preprint]_.
The third example demonstrates feature consistency analysis for model selection for the case of non-stationary data. It reproduces Figure 5a-c in the main text [#Genkin2020preprint]_.

.. toctree::
    :maxdepth: 2

    examples/2021_learning_nonstationary_dynamics/Example1.ipynb
    examples/2021_learning_nonstationary_dynamics/Example2.ipynb
    examples/2021_learning_nonstationary_dynamics/Example3.ipynb
    
*********************************************************************************************
The dynamics and geometry of choice in premotor cortex
*********************************************************************************************

The first example fits single-neuron model from PMd data and selects an optimal
model using feature consistency analysis, see Figure 3 in [#Genkin2023]_.
The second example fits population model from PMd data and selects an optimal
model using feature consistency analysis, see Figure 4 in [#Genkin2023]_.

.. toctree::
    :maxdepth: 2

    examples/2024_the_dynamics_and_geomerty/Example1.ipynb
    examples/2024_the_dynamics_and_geomerty/Example2.ipynb
    
    
References
----------

.. [#Genkin2020] `Genkin, M., Engel, T.A. Moving beyond generalization to accurate interpretation of flexible models. Nat Mach Intell 2, 674–683 (2020). <https://www.nature.com/articles/s42256-020-00242-6>`_

.. [#Genkin2021] `Genkin, M., Hughes, O. and Engel, T.A. Learning non-stationary Langevin dynamics from stochastic observations of latent trajectories. Nat Commun 12, 5986 (2021). <https://www.nature.com/articles/s41467-021-26202-1>`_

.. [#Genkin2023] `Genkin M, Shenoy KV, Chandrasekaran C, Engel TA. The dynamics and geometry of choice in premotor cortex. bioRxiv [Preprint] (2023) . <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10401920>`_