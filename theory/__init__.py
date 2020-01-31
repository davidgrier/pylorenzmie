import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from .Particle import Particle
from .Instrument import Instrument, coordinates

try:
    from .CudaGeneralizedLorenzMie \
        import CudaGeneralizedLorenzMie as GeneralizedLorenzMie
    from .FastSphere import FastSphere as Sphere
except Exception as e:
    logger.info("Could not import CUDA GPU pipeline. "
                + str(e))
    try:
        from .FastGeneralizedLorenzMie \
            import FastGeneralizedLorenzMie as GeneralizedLorenzMie
        from .FastSphere import FastSphere as Sphere
    except Exception as e:
        logger.info(
            "Could not import numba CPU pipeline. "
            + str(e))
        from .GeneralizedLorenzMie import GeneralizedLorenzMie
        from .Sphere import Sphere

from .LorenzMie import LorenzMie
from .LMHologram import LMHologram
from .Feature import Feature


all = [GeneralizedLorenzMie, Sphere, Feature,
       LMHologram, Instrument, coordinates, Particle, LorenzMie]

