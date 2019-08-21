import cython
from scipy.optimize import OptimizeResult
import numpy as np
cimport numpy as np

np.import_array()

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


@cython.boundscheck(False)
@cython.wraparound(False)
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
    cdef np.ndarray[DTYPE_t, ndim = 2] simplex = np.zeros([N+1, N])
    simplex[0] = x0
    for i in range(N):
        simplex[i+1] = x0
        simplex[i+1, i] += simplex_scale[i]
    # Initialize algorithm
    cdef int max_nfev = maxevals
    cdef int neval = 1
    cdef int niter = 1
    cdef np.ndarray one2np1 = np.arange(1, N+1)
    cdef np.ndarray evals = np.zeros(N+1)
    cdef np.ndarray idxs = np.zeros(N+1)
    for idx in range(N+1):
        simplex[idx] = np.maximum(xmin, np.minimum(simplex[idx], xmax))
        evals[idx] = objective(simplex[idx])
        neval += 1
    idxs = np.argsort(evals)
    evals = np.take(evals, idxs, 0)
    simplex = np.take(simplex, idxs, 0)

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
    message = 'failure (hit max evals)'
    while(neval < max_nfev):
        # Test if simplex is small
        if all(np.amax(np.abs(simplex[1:] - simplex[0]), axis=0) <= xtol):
            message = 'convergence (simplex small)'
            break
        # Test if function values are similar
        if np.max(np.abs(simplex[0] - simplex[1:])) <= ftol:
            message = 'convergence (fvals similar)'
        # Test if simplex hits edge of parameter space
        end = False
        for k in range(N):
            temp = simplex[:, k]
            if xmax[k] in temp or xmin[k] in temp:
                end = True
        if end:
            message = 'failure (stuck to boundary)'
            break
        # Reflect
        xbar = np.add.reduce(simplex[:N], 0) / N
        xr = (1 + rho) * xbar - rho * simplex[N]
        xr = np.maximum(xmin, np.minimum(xr, xmax))
        fxr = objective(xr)
        neval += 1
        doshrink = 0
        # Check if reflection is better than best estimate
        if fxr < evals[0]:
            # If so, reflect double and see if that's even better
            xe = (1 + rho * chi) * xbar - rho * chi * simplex[N]
            xe = np.maximum(xmin, np.minimum(xe, xmax))
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
                    xc = (1 + psi * rho) * xbar - psi * rho * simplex[N]
                    xc = np.maximum(xmin, np.minimum(xc, xmax))
                    fxc = objective(xc)
                    neval += 1
                    if fxc <= fxr:
                        simplex[N] = xc
                        evals[N] = fxc
                    else:
                        doshrink = 1
                else:
                    # Do 'inside' contraction
                    xcc = (1 - psi) * xbar + psi * simplex[N]
                    xcc = np.maximum(xmin, np.minimum(xcc, xmax))
                    fxcc = objective(xcc)
                    neval += 1
                    if fxcc < evals[N]:
                        simplex[N] = xcc
                        evals[N] = fxcc
                    else:
                        doshrink = 1
                if doshrink:
                    for j in one2np1:
                        simplex[j] = simplex[0] + sigma * \
                            (simplex[j] - simplex[0])
                        simplex[j] = np.maximum(
                            xmin, np.minimum(simplex[j], xmax))
                        evals[j] = objective(simplex[j])
                        neval += 1
        idxs = np.argsort(evals)
        simplex = np.take(simplex, idxs, 0)
        evals = np.take(evals, idxs, 0)
        niter += 1
    best = simplex[0]
    chi = evals[0]
    success = False if 'failure' in message else True
    return OptimizeResult(x=best, success=success, message=message,
                          nit=niter, nfev=neval, fun=chi)
