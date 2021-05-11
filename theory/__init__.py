#import pylorenzmie.utilities.configuration as config
import sys

from .Particle import Particle
from .Sphere import Sphere
from .Instrument import Instrument

#from .LorenzMie import LorenzMie as numpyLorenzMie
#if config.has_cupy():
#    from .cupyLorenzMie import cupyLorenzMie as cupyLorenzMie
#    LorenzMie = cupyLorenzMie
#else:
#    LorenzMie = numpyLorenzMie
if 'cupy' in sys.modules:
    from .cupyLorenzMie import cupyLorenzMie as LorenzMie
else:
    from .LorenzMie import LorenzMie

from .LMHologram import LMHologram

__all__ = [Particle, Sphere, Instrument, LorenzMie, LMHologram]
