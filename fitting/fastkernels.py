from numba import njit, prange

# See https://llvm.org/docs/LangRef.html#fast-math-flags
# for a list of fastmath flags for LLVM compiler
safe_flags = {'nnan', 'ninf', 'arcp', 'nsz'}


@njit(parallel=True, fastmath=False, cache=True)
def fastresiduals(holo, data, noise):
    return (holo - data) / noise


@njit(parallel=True, fastmath=False, cache=True)
def fastchisqr(holo, data, noise):
    chisqr = 0.
    for idx in prange(holo.size):
        chisqr += ((holo[idx] - data[idx]) / noise) ** 2
    return chisqr


@njit(parallel=True, fastmath=False, cache=True)
def fastabsolute(holo, data, noise):
    s = 0.
    for idx in prange(holo.size):
        s += abs((holo[idx] - data[idx]) / noise)
    return s
