# Configuration for optional components of pylorenzmie
use_numba = True
use_cupy = True


#
import logging
logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.WARNING)

try:
    import numba
    import_numba = True
except ImportError as numba_ex:
    import_numba = False
    logger.warn('Cannot import numba:\n\t{}'.format(numba_ex))


def has_numba():
    if not use_numba:
        logger.warn('Numba deselected in {}'.format(__file__))
    #if not import_numba:
        #logger.warn('Cannot import numba:\n\t{}'.format(numba_ex))
    ok = use_numba and import_numba
    if not ok:
        logger.warn('Falling back to standard implementation')
    return ok


try:
    import cupy
    import_cupy = True
except ImportError as cupy_ex:
    import_cupy = False
    logger.warn('Cannot import cupy:\n\t{}'.format(cupy_ex))


def has_cupy():
    if not use_cupy:
        logger.warn('cupy deselected in {}'.format(__file__))
    #if not import_cupy:
        #logger.warn('Cannot import cupy:\n\t{}'.format(cupy_ex))
    ok = use_cupy and import_cupy
    if not ok:
        logger.warn('Falling back to standard implementation')
    return ok

    
