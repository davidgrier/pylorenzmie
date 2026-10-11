import logging as _logging
import os as _os
from typing import TYPE_CHECKING as _TYPE_CHECKING

from .Particle import Particle
from .Sphere import Sphere
from .Cluster import Cluster
from .Dimer import Dimer
from .Instrument import Instrument
from .Aberrated import (Aberrated, AberratedLorenzMie)
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
    raise ValueError(f'Unknown backend {name!r}: choose numpy, numba, '
                     'torch, jax or cupy')


def _select_backend(requested: str | None = None) -> type:
    requested = (requested or '').strip().lower()
    if requested:
        try:
            return _load(requested)
        except ValueError:
            raise
        except Exception as error:
            raise ImportError(
                f'PYLORENZMIE_BACKEND={requested!r} is not available: '
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

# Alias for backward compatibility
LMHologram = LorenzMie

__all__ = '''Instrument Particle Sphere Cluster Dimer
          LorenzMie LMHologram Aberrated AberratedLorenzMie'''.split()
