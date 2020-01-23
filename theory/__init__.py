import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
try:
    from pylorenzmie.theory.CudaGeneralizedLorenzMie \
        import CudaGeneralizedLorenzMie as GeneralizedLorenzMie
    from pylorenzmie.theory.FastSphere import FastSphere as Sphere
except Exception as e:
    logger.info("Could not import CUDA GPU pipeline. "
                + str(e))
    try:
        from pylorenzmie.theory.FastGeneralizedLorenzMie \
            import FastGeneralizedLorenzMie as GeneralizedLorenzMie
        from pylorenzmie.theory.FastSphere import FastSphere as Sphere
    except Exception as e:
        logger.info(
            "Could not import numba CPU pipeline. "
            + str(e))
        from pylorenzmie.theory.GeneralizedLorenzMie \
            import GeneralizedLorenzMie
        from pylorenzmie.theory.Sphere import Sphere

from pylorenzmie.theory.Video import Video
from pylorenzmie.theory.Frame import Frame
from pylorenzmie.theory.Feature import Feature
from pylorenzmie.theory.LMHologram import LMHologram
from pylorenzmie.theory.Instrument import Instrument, coordinates
from pylorenzmie.theory.Particle import Particle
from pylorenzmie.theory.LorenzMie import LorenzMie
from pylorenzmie.theory.Trajectory import Trajectory

all = [GeneralizedLorenzMie, Sphere, Video, Frame, Feature,
       LMHologram, Instrument, coordinates, Particle, LorenzMie, Trajectory]
