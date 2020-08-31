.. _installation:

Installation
============

Package only
------------
The package can be installed from git repository with pip. ::

    pip install git+https://github.com/engellab/neuralflow
    
    
Package and examples
--------------------
To get the package with the examples, one needs to clone the repository. The examples are provided as Jupyter notebook 
(ipynb) files, so ``jupyter`` package has to be preinstalled. With conda as a package manager, one may opt to 
create a new environment::

     conda create -n neuralflow jupyter pip && conda activate neuralflow
    
Alternatively, one can work in the base environment (make sure that ``jupyter`` package is installed). 
Clone the repository and go to the repository root directory::

     git clone https://github.com/engellab/neuralflow
     cd neuralflow
    
Install the package from a local copy::

    pip install .
    
After that, you should be able to run ``example/Example1.ipynb`` and ``example/Example2.ipynb``

Install C extensions from pyx files
--------------------------
By default, C extensions are installed from .c file(s). These .c files were generated from .pyx files with cython.
If you want to install the package's extensions from the original pyx files, change line 6 in setup.py to ``USE_CYTHON = 1``. In this case, 
`cython` package is required for the installation (not included in the requirements list, so you will need to run ``pip install cython``).
 