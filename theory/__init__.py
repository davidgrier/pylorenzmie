import pylorenzmie.utilities.configuration as config

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

from .Particle import Particle
from .Sphere import Sphere

from .Instrument import (Instrument, coordinates)

try:
    if not config.use_cupy:
        raise ImportError('Cupy deselected in {}'.format(config.__file__))
    from .cupyLorenzMie import cupyLorenzMie as LorenzMie
except ImportError as ex:
    logger.warn('Cannot import cupyLorenzMie:' +
                '\n\t{}'.format(ex) +
                '\n\tFalling back to LorenzMie')
    from .LorenzMie import LorenzMie

from .LMHologram import LMHologram

__all__ = [Particle, Sphere, Instrument, coordinates,
           LorenzMie, LMHologram]
