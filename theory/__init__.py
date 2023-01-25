import sys

from .Particle import Particle
from .Sphere import Sphere
from .Cluster import Cluster
from .Dimer import Dimer
from .Instrument import Instrument

if 'cupy' in sys.modules:
    from .LorenzMie import LorenzMie as numpyLorenzMie
    from .cupyLorenzMie import cupyLorenzMie as LorenzMie
else:
    from .LorenzMie import LorenzMie
    numpyLorenzMie = LorenzMie
from .AberratedLorenzMie import AberratedLorenzMie

# Alias for backward compatibility
LMHologram = LorenzMie

__all__ = 'Instrument Particle Sphere Cluster Dimer' \
          'LorenzMie AberratedLorenzMie LMHologram'.split()
