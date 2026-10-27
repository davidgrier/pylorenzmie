'''Names of the calculators of the scattering model.

Kept apart from :mod:`pylorenzmie.theory` so that the command line of
``lmtool`` can offer the choices without importing the whole package.
'''

BACKENDS = ('numpy', 'numba', 'torch', 'jax', 'cupy')
