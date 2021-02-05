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


    
