import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
try:
    bloop()
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

all = [GeneralizedLorenzMie, Sphere]
