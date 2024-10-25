from setuptools import setup, dist
from setuptools.extension import Extension
#from distutils.extension import Extension
#dist.Distribution().fetch_build_eggs(['numpy'])
import numpy as numpy

USE_CYTHON = 1   # change to 0 to build the extension from c.

ext = '.pyx' if USE_CYTHON else '.c'


extensions = [
    Extension(
        "neuralflow.c_get_gamma", 
        ["neuralflow/c_get_gamma" + ext], 
        include_dirs=[numpy.get_include(), "neuralflow/"]
        )
    ]

if USE_CYTHON:
    from Cython.Build import cythonize
    extensions = cythonize(extensions, build_dir="neuralflow/build")


setup(name='neuralflow',
      description='Modeling neural spiking activity with a contnuous latent Langevin dynamics',
      version='3.0.0',
      ext_modules=extensions,
      packages=["neuralflow", "neuralflow.utilities"],
      keywords='Neuroscience, Machine learning, Langevin modeling',
      author='Mikhail Genkin and Tatiana A. Engel',
      author_email='engel@cshl.edu',
      license='MIT',
      include_package_data=True,
      setup_requires=['numpy'],
      install_requires=[
          'numpy',
          'matplotlib',
          'pandas',
          'scipy',
          'tqdm',
          'scikit-learn',
      ],
      zip_safe=False)
