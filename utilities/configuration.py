from importlib import import_module
import logging

use_cupy = True
use_numba = True
use_catch = True

logging.basicConfig()
logger = logging.getLogger('configuration')
logger.setLevel(logging.INFO)


def has_(module):
    '''Return True if module is selected and can be imported'''
    selected = globals()['use_'+module.lower()]
    can_import = False
    if selected:
        try:
            import_module(module)
            can_import = True
        except ImportError as ex:
            logger.warn(f' Cannot import {module}:\n\t{ex}')
    else:
        logger.info(f' {module} deselected in {__file__}')
    return selected and can_import


has_cupy = lambda: has_('cupy')
has_numba = lambda: has_('numba')
has_catch = lambda: has_('CATCH')
