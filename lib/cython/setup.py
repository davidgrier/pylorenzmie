from distutils.core import setup
from Cython.Build import cythonize
import numpy

'''Run command "python3 setup.py build_ext --inplace" for compilation'''

setup(
    ext_modules=cythonize("cminimizers.pyx",
                          language_level='3',
                          annotate=True),
    include_dirs=[numpy.get_include()]
)
