.. _installation:

Installation
============

Package only
------------
The package can be installed from a Git repository with pip. ::

    pip install git+https://github.com/engellab/neuralflow
    
    
Package and examples
--------------------
To get the package with the examples, one needs to clone the repository. The examples are provided as Jupyter notebook 
(ipynb) files, so ``jupyter`` package has to be preinstalled. With Conda as a package manager, one may opt to
create a new environment::

     conda create -n neuralflow jupyter pip && conda activate neuralflow
    
Alternatively, one can work in the base environment (make sure that ``jupyter`` package is installed). 
Clone the repository and go to the repository root directory::

     git clone https://github.com/engellab/neuralflow
     cd neuralflow
    
Install the package from a local copy::

    pip install .
    
After that, you should be able to run examples in ``examples`` folder.
If you have issues with the Cython extension and want to use a precompiled .c file instead, open setup.py and change line 7 to USE_CYTHON = 0


CUDA support
------------

Optimization can be performed on CUDA-enabled GPU. For GPU support, cupy
package has to be installed on a machine with CUDA-enabled GPU. The package
was tested with cupy version 12.2.0. Note that double-precision computations
are absolutely necessary for our framework, so optimization benefits from
GPU acceleration only if scientific grade GPU (e.g. Tesla V100) is used. A
gaming GPU performance is approximately the same as CPU performance, since
gaming GPUs do not have many double-precision multiprocessors.