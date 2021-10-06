from importlib import import_module
import logging


use_numba = True
use_catch = True

logging.basicConfig()
logger = logging.getLogger('configuration')
logger.setLevel(logging.INFO)


def has_(module):
    '''Return True if module is selected and can be imported'''
    flag = globals()['use_'+module.lower()] or False
    can_import = False
    if flag:
        try:
            import_module(module)
            can_import = True
        except ImportError as ex:
            logger.warn(' Cannot import {}:\n\t{}'.format(module, ex))
    else:
        logger.info(' {} deselected in {}'.format(module, __file__))
    ok = flag and can_import
    if not ok:
        logger.info(' Falling back to standard implementation')
    return ok


has_numba = lambda: has_('numba')
has_catch = lambda: has_('CATCH')
