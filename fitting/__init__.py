import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

try:
    from .cython.cminimizers import amoeba
except Exception as e:
    logger.info(e)
    from .minimizers import amoeba

from .Settings import FitSettings, FitResult
from .Mask import Mask
from .GlobalSampler import GlobalSampler
from .Optimizer import Optimizer


all = ['mie_fit.py', 'minimizers.py', amoeba, GlobalSampler,
       FitSettings, FitResult, Mask, Optimizer]
