# Use Numba if available. Otherwise fall back to standard interpreter

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

try:
    import numba

    jit = numba.jit
    njit = numba.njit
    prange = numba.prange

    logger.debug('Numba compiler successfully imported')
    
except ImportError:
    logger.warn('Cannot import Numba compiler: ' +
                '\n\tFalling back to standard interpreter')

    def null_decorator(pyfunc=None, **kwargs):
        '''Null decorator if Numba accelerators are not available'''
        def wrap(func):
            return func
        return wrap if pyfunc is None else wrap(pyfunc)

    jit = null_decorator
    njit = null_decorator

    def prange(*args):
        '''fall back to standard range'''
        return range(*args)


    
