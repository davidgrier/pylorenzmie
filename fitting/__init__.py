import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

try:
    from .cython.cminimizers import amoeba
except Exception as e:
    logger.info(e)
    from .minimizers import amoeba

from .Settings import (Parameter, FitSettings, FitResult)
from .Mask import Mask
from .Optimizer import Optimizer


__all__ = [amoeba, Parameter, FitSettings, FitResult, Mask, Optimizer]
