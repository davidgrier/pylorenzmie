import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
try:
    from pylorenzmie.theory.cudaGeneralizedLorenzMie \
        import GeneralizedLorenzMie
    from pylorenzmie.theory.fastSphere import Sphere
except Exception as e:
    logger.info("Could not import CUDA GPU pipeline. "
                + str(e))
    try:
        from pylorenzmie.theory.fastGeneralizedLorenzMie \
            import GeneralizedLorenzMie
        from pylorenzmie.theory.fastSphere import Sphere
    except Exception as e:
        logger.info(
            "Could not import numba CPU pipeline. "
            + str(e))
        from pylorenzmie.theory.GeneralizedLorenzMie \
            import GeneralizedLorenzMie
        from pylorenzmie.theory.Sphere import Sphere

all = [GeneralizedLorenzMie, Sphere]
