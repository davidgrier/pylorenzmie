import logging as _logging

from .Particle import Particle
from .Sphere import Sphere
from .Cluster import Cluster
from .Dimer import Dimer
from .Instrument import Instrument
from .Aberrated import (Aberrated, AberratedLorenzMie)
from .LorenzMie import LorenzMie as _BaseLorenzMie

# Select the best available backend in priority order: JAX > CuPy > Numba > NumPy.
# Each accelerated module raises ImportError (missing library) or a backend-
# specific exception (e.g. StableHLO version mismatch for jax-metal) on import.
# ImportErrors are silent; other failures are logged as warnings.
LorenzMie = _BaseLorenzMie

try:
    from .jaxLorenzMie import jaxLorenzMie
    LorenzMie = jaxLorenzMie
except ImportError:
    pass
except Exception as _e:
    _logging.getLogger(__name__).warning(
        'JAX backend unavailable (%s); trying next backend.', _e)

if LorenzMie is _BaseLorenzMie:
    try:
        from .cupyLorenzMie import cupyLorenzMie
        LorenzMie = cupyLorenzMie
    except Exception:
        pass

if LorenzMie is _BaseLorenzMie:
    try:
        from .numbaLorenzMie import numbaLorenzMie
        LorenzMie = numbaLorenzMie
    except Exception:
        pass

# Alias for backward compatibility
LMHologram = LorenzMie

__all__ = '''Instrument Particle Sphere Cluster Dimer
          LorenzMie LMHologram Aberrated AberratedLorenzMie'''.split()
