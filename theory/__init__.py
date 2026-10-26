import logging as _logging
import os as _os
from typing import TYPE_CHECKING as _TYPE_CHECKING

from .Particle import Particle
from .Sphere import Sphere
from .Cluster import Cluster
from .Dimer import Dimer
from .Pair import Pair
from .Instrument import Instrument
from .Aberrated import Aberrated
from .Aberrated import AberratedLorenzMie as _NumpyAberratedLorenzMie
from .Distorted import Distorted, supported as _distortable
from .Distorted import DistortedLorenzMie as _NumpyDistortedLorenzMie
from .LorenzMie import LorenzMie as _BaseLorenzMie

# Choice of the calculator exported as ``LorenzMie``.
#
# By default it is the fastest backend that needs no special hardware:
# Numba if it is installed and works, otherwise NumPy.  The GPU backends
# (torch, CuPy) and JAX are not selected automatically: a GPU pays off for
# large images and many pixels, not for the small masked fits that are
# typical, and the speed depends on the hardware.  On a CPU the Numba kernel
# is several times faster than JAX for one sphere and much faster for
# clusters (which JAX hands back to NumPy) and for fits (JAX recompiles
# whenever the number of partial waves changes).  Choose a backend with the
# environment variable PYLORENZMIE_BACKEND (numpy, numba, torch, jax or
# cupy), or import the class from its module, e.g.
# ``from pylorenzmie.theory.torchLorenzMie import TorchLorenzMie``.
# A backend that is requested explicitly but cannot be loaded raises an error
# rather than silently falling back to a different one.


BACKENDS = ('numpy', 'numba', 'torch', 'jax', 'cupy')


def _load(name: str) -> type:
    if name == 'numpy':
        return _BaseLorenzMie
    if name == 'numba':
        from .numbaLorenzMie import numbaLorenzMie
        return numbaLorenzMie
    if name == 'jax':
        from .jaxLorenzMie import jaxLorenzMie
        return jaxLorenzMie
    if name == 'torch':
        from .torchLorenzMie import TorchLorenzMie
        return TorchLorenzMie
    if name == 'cupy':
        from .cupyLorenzMie import cupyLorenzMie
        return cupyLorenzMie
    raise ValueError(f'Unknown backend {name!r}: choose from '
                     + ', '.join(BACKENDS))


def _select_backend(requested: str | None = None) -> type:
    requested = (requested or '').strip().lower()
    if requested:
        try:
            return _load(requested)
        except ValueError:
            raise
        except Exception as error:
            raise ImportError(
                f'The {requested!r} backend is not available: '
                f'{error}') from error
    try:
        return _load('numba')
    except ImportError:
        return _BaseLorenzMie
    except Exception as error:
        _logging.getLogger(__name__).warning(
            'Numba backend unavailable (%s); using NumPy.', error)
        return _BaseLorenzMie


if _TYPE_CHECKING:
    # the calculator is chosen at run time, but type checkers need a class
    LorenzMie = _BaseLorenzMie
else:
    LorenzMie = _select_backend(_os.environ.get('PYLORENZMIE_BACKEND'))



def _aberrated(base: type) -> type:
    '''The aberrated calculator for a backend.

    The NumPy one is defined in :mod:`~pylorenzmie.theory.Aberrated`; the
    others are made here, so that the class can be pickled by name.
    '''
    if base is _BaseLorenzMie:
        return _NumpyAberratedLorenzMie
    aberrated = Aberrated(base)
    aberrated.__module__ = __name__
    aberrated.__qualname__ = aberrated.__name__ = 'AberratedLorenzMie'
    return aberrated


if _TYPE_CHECKING:
    AberratedLorenzMie = _NumpyAberratedLorenzMie
else:
    # ``AberratedLorenzMie`` follows the backend, like ``LorenzMie``; the
    # NumPy calculator is ``pylorenzmie.theory.Aberrated.AberratedLorenzMie``
    AberratedLorenzMie = _aberrated(LorenzMie)


def _distorted(base: type) -> type:
    '''The distorted calculator for a backend.

    Numba and NumPy are supported.  The GPU and JAX calculators do not
    evaluate the field from displacements, so they get the NumPy one.
    '''
    if base is _BaseLorenzMie or not _distortable(base):
        return _NumpyDistortedLorenzMie
    distorted = Distorted(base)
    distorted.__module__ = __name__
    distorted.__qualname__ = distorted.__name__ = 'DistortedLorenzMie'
    return distorted


if _TYPE_CHECKING:
    DistortedLorenzMie = _NumpyDistortedLorenzMie
else:
    # follows the backend where it can; see :func:`Distorted`
    DistortedLorenzMie = _distorted(LorenzMie)


def model_class(backend: str | None = None, distorted: bool = False) -> type:
    '''The microscope model for a back end.

    Parameters
    ----------
    backend : str, optional
        One of ``'numpy'``, ``'numba'``, ``'torch'``, ``'jax'`` or
        ``'cupy'``.  Default: the calculator that this package exports as
        :class:`LorenzMie`, which ``PYLORENZMIE_BACKEND`` chooses.
    distorted : bool, optional
        If True, return the :func:`Distorted` calculator, with the
        aberrations as geometric distortions.  Otherwise return the
        :func:`Aberrated` calculator, with the aberrations as phase
        factors.  Default: False.

    Returns
    -------
    type
        A subclass of the calculator of the back end.  Asking again for the
        same back end returns the same class, which pickle can find by its
        name.

    Raises
    ------
    ValueError
        If *backend* is not the name of a back end, or if *distorted* is
        True for a back end that cannot evaluate the field at warped
        coordinates (only ``'numpy'`` and ``'numba'`` can).
    ImportError
        If the back end is not available.
    '''
    base = _select_backend(backend) if backend else LorenzMie
    if distorted and not _distortable(base):
        raise ValueError(f'Distorted does not support {base.__name__}: '
                         'choose the numpy or numba backend')
    if base is _BaseLorenzMie:
        return _NumpyDistortedLorenzMie if distorted \
            else _NumpyAberratedLorenzMie
    if base is LorenzMie:
        return DistortedLorenzMie if distorted else AberratedLorenzMie
    factory, prefix = (Distorted, 'Distorted') if distorted \
        else (Aberrated, 'Aberrated')
    name = prefix + base.__name__
    if name not in globals():
        cls = factory(base)
        # named so that pickle finds the class in this module
        cls.__module__ = __name__
        cls.__qualname__ = cls.__name__ = name
        globals()[name] = cls
    return globals()[name]


# Alias for backward compatibility
LMHologram = LorenzMie

__all__ = '''Instrument Particle Sphere Cluster Dimer Pair
          LorenzMie LMHologram Aberrated AberratedLorenzMie
          Distorted DistortedLorenzMie model_class BACKENDS'''.split()
