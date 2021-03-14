# Configuration for optional components of pylorenzmie
use_numba = True
use_cupy = True
use_catch = True


# Implement requested configuration
import logging
logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.WARNING)

try:
    import numba
    import_numba = True
except ImportError as ex:
    numba_ex = ex
    import_numba = False


def has_numba():
    if not use_numba:
        logger.warn(' numba deselected in {}'.format(__file__))
    elif not import_numba:
        logger.warn(' Cannot import numba:\n\t{}'.format(numba_ex))
    ok = use_numba and import_numba
    if not ok:
        logger.warn(' Falling back to standard implementation')
    return ok


try:
    import cupy
    import_cupy = True
except ImportError as ex:
    cupy_ex = ex
    import_cupy = False


def has_cupy():
    if not use_cupy:
        logger.warn(' cupy deselected in {}'.format(__file__))
    elif not import_cupy:
        logger.warn(' Cannot import cupy:\n\t{}'.format(cupy_ex))
    ok = use_cupy and import_cupy
    if not ok:
        logger.warn(' Falling back to standard implementation')
    return ok


'''
try:
    import CATCH
    import_catch = True
except ImportError as ex:
    catch_ex = ex
    import_catch = False


def has_catch():
    if not use_catch:
        logger.warn(' CATCH deselected in {}'.format(__file__))
    elif not import_catch:
        logger.warn(' Cannot import CATCH:\n\t{}'.format(catch_ex))
    ok = use_catch and import_catch
    if not ok:
        logger.warn(' Falling back to standard implementation')
    return ok
'''
