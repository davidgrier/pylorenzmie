import cython
from cython.parallel import prange
from scipy.optimize import OptimizeResult
import numpy as np
cimport numpy as np

np.import_array()

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def amoeba(objective,
           np.ndarray[DTYPE_t, ndim=1] x0,
           np.ndarray[DTYPE_t, ndim=1] xmin,
           np.ndarray[DTYPE_t, ndim=1] xmax,
           np.ndarray[DTYPE_t, ndim=1] simplex_scale,
           np.ndarray[DTYPE_t, ndim=1] xtol,
           initial_simplex=None, maxevals=int(1e3), ftol=1e-7):
    '''Nelder-mead optimization adapted from scipy.optimize.fmin'''
    # Initialize simplex
    cdef int N = len(x0)
    cdef int i, j
    cdef np.ndarray[DTYPE_t, ndim = 2] simplex = np.zeros([N+1, N])
    simplex[0] = x0
    for i in range(N):
        simplex[i+1] = x0
        simplex[i+1, i] += simplex_scale[i]
    # Initialize algorithm
    cdef int max_nfev = maxevals
    cdef double fcntol = ftol
    cdef int neval = 1
    cdef int niter = 1
    cdef np.ndarray[DTYPE_t, ndim = 1] evals = np.zeros(N+1)
    cdef np.ndarray[np.int_t, ndim = 1] idxs = np.zeros(N+1,
                                                       dtype=np.int)
    for i in range(N+1):
        simplex[i] = maximum(xmin, minimum(simplex[i], xmax))
        evals[i] = objective(simplex[i])
        neval += 1
    idxs = sort(evals)
    #evals = take1d(evals, idxs)
    simplex = take2d(simplex, idxs)

    cdef double rho = 1.
    cdef double chi = 2.
    cdef double psi = 0.5
    cdef double sigma = 0.5

    cdef np.ndarray[DTYPE_t, ndim = 1] xbar = np.zeros(N)
    cdef np.ndarray[DTYPE_t, ndim = 1] xr = np.zeros(N)
    cdef np.ndarray[DTYPE_t, ndim = 1] xe = np.zeros(N)
    cdef np.ndarray[DTYPE_t, ndim = 1] xc = np.zeros(N)
    cdef np.ndarray[DTYPE_t, ndim = 1] xcc = np.zeros(N)

    # START FITTING
    cdef str message = 'failure (hit max evals)'
    cdef bint end
    cdef np.ndarray[DTYPE_t, ndim = 1] temp
    cdef int doshrink
    cdef DTYPE_t s

    while(neval < max_nfev):
        # Test if simplex is small
        if all(np.amax(np.abs(simplex[1:] - simplex[0]), axis=0) <= xtol):
            message = 'convergence (simplex small)'
            break
        # Test if function values are similar
        if np.max(np.abs(simplex[0] - simplex[1:])) <= fcntol:
            message = 'convergence (fvals similar)'
        # Test if simplex hits edge of parameter space
        end = False
        for i in range(N):
            temp = simplex[:, i]
            if xmax[i] in temp or xmin[i] in temp:
                end = True
        if end:
            message = 'failure (stuck to boundary)'
            break
        # Reflect of array
        for i in range(N):
            s = 0.
            temp = simplex[:, i]
            for j in range(N):
                s += temp[j]
            xbar[i] = s / N
        for i in range(N):
            xr[i] = (1 + rho) * xbar[i] - rho * simplex[N, i]
        xr = maximum(xmin, minimum(xr, xmax))
        fxr = objective(xr)
        neval += 1
        doshrink = 0
        # Check if reflection is better than best estimate
        if fxr < evals[0]:
            # If so, reflect double and see if that's even better
            for i in range(N):
                xe[i] = (1. + rho * chi) * xbar[i]\
                    - rho * chi * simplex[N, i]
            xe = maximum(xmin, minimum(xe, xmax))
            fxe = objective(xe)
            neval += 1
            if fxe < fxr:
                simplex[N] = xe
                evals[N] = fxe
            else:
                simplex[N] = xr
                evals[N] = fxr
        else:
            if fxr < evals[N-1]:
                simplex[N] = xr
                evals[N] = fxr
            else:
                # If reflection is not better, contract.
                if fxr < evals[N]:
                    for i in range(N):
                        xc[i] = (1. + psi * rho) * xbar[i] \
                            - psi * rho * simplex[N, i]
                    xc = maximum(xmin, minimum(xc, xmax))
                    fxc = objective(xc)
                    neval += 1
                    if fxc <= fxr:
                        simplex[N] = xc
                        evals[N] = fxc
                    else:
                        doshrink = 1
                else:
                    # Do 'inside' contraction
                    for i in range(N):
                        xcc[i] = (1 - psi) * xbar[i] \
                            + psi * simplex[N, i]
                    xcc = maximum(xmin, minimum(xcc, xmax))
                    fxcc = objective(xcc)
                    neval += 1
                    if fxcc < evals[N]:
                        simplex[N] = xcc
                        evals[N] = fxcc
                    else:
                        doshrink = 1
                if doshrink:
                    for i in range(1, N+1):
                        for j in range(N):
                            simplex[i, j] = simplex[0, j] + sigma * \
                                (simplex[i, j] - simplex[0, j])
                        simplex[i] = maximum(
                            xmin, minimum(simplex[i], xmax))
                        evals[i] = objective(simplex[i])
                        neval += 1
        idxs = sort(evals)
        simplex = take2d(simplex, idxs)
        #evals = take1d(evals, idxs)
        niter += 1
    cdef bint success = False if 'failure' in message else True
    return OptimizeResult(x=simplex[0],
                          success=success,
                          message=message,
                          nit=niter,
                          nfev=neval,
                          fun=evals[0])


cdef np.ndarray take1d(np.ndarray[DTYPE_t, ndim=1] a,
                       np.ndarray[np.int_t, ndim=1] idxs):
    cdef int n = a.shape[0]
    cdef np.ndarray[DTYPE_t, ndim = 1] b = np.zeros(n)
    cdef int i
    for i in range(n):
        b[i] = a[idxs[i]]
    return b

cdef np.ndarray take2d(np.ndarray[DTYPE_t, ndim=2] a,
                       np.ndarray[np.int_t, ndim=1] idxs):
    cdef int nx, ny
    nx = a.shape[0]
    ny = a.shape[1]
    cdef np.ndarray[DTYPE_t, ndim = 2] b = np.zeros((nx, ny))
    cdef int i
    for i in range(nx):
        b[i] = a[idxs[i]]
    return b

cdef np.ndarray minimum(np.ndarray[DTYPE_t, ndim=1] a1,
                        np.ndarray[DTYPE_t, ndim=1] a2):
    cdef int n = a1.shape[0]
    cdef np.ndarray[DTYPE_t, ndim = 1] b = np.zeros(n)
    cdef DTYPE_t m
    cdef int i
    for i in range(n):
        if a1[i] < a2[i]:
            m = a1[i]
        else:
            m = a2[i]
        b[i] = m
    return b

cdef np.ndarray maximum(np.ndarray[DTYPE_t, ndim=1] a1,
                        np.ndarray[DTYPE_t, ndim=1] a2):
    cdef int n = a1.shape[0]
    cdef np.ndarray[DTYPE_t, ndim = 1] b = np.zeros(n)
    cdef DTYPE_t m
    cdef int i
    for i in range(n):
        if a1[i] > a2[i]:
            m = a1[i]
        else:
            m = a2[i]
        b[i] = m
    return b

cdef np.ndarray sort(np.ndarray[DTYPE_t, ndim=1] a):
    cdef int m = a.shape[0]
    cdef int i, j
    cdef DTYPE_t key
    cdef np.ndarray[np.int_t, ndim= 1] order = np.zeros(m,
                                                         dtype=np.int)
    for i in range(m):
        order[i] = i
    # Traverse through 1 to len(arr)
    for i in range(1, m):
        key = a[i]
        # Move elements of a[0..i-1], that are
        # greater than key, to one position ahead
        # of their current position
        j = i-1
        while j >= 0 and key < a[j]:
            a[j+1] = a[j]
            order[j+1] = order[j]
            j -= 1
        a[j+1] = key
        order[j+1] = i
    return order
