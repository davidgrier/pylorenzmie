'''Azimuthal statistics for 2-D images.

The implementation lives in `dgimage <https://dgimage.readthedocs.io>`_, the
library of image transformations that pylorenzmie, QVideo and other projects of
the group share; this module keeps the names that pylorenzmie has always
offered: :func:`avg`, :func:`std`, :func:`med`, :func:`mad` and
:func:`azimuthaloperator`.  See :mod:`dgimage.azimuthal`.
'''

from dgimage.azimuthal import avg, azimuthaloperator, mad, med, std

__all__ = ['azimuthaloperator', 'avg', 'std', 'med', 'mad']
