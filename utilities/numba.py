def jit(pyfunc=None, **kwargs):
    def wrap(func):
        return func

    return wrap if pyfunc is None else wrap(pyfunc)

def njit(pyfunc=None, **kwargs):
    def wrap(func):
        return func

    return wrap if pyfunc is None else wrap(pyfunc)

def prange(*args):
    return range(*args)


    
