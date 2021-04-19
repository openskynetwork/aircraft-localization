from distutils.core import Extension, setup
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(
        "optimize",
        ["src/optimize.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    )
]

setup(
    name = 'Optimization functions for solving multilateral equations',
    ext_modules = cythonize(ext_modules),
    include_dirs=[numpy.get_include()]
    )