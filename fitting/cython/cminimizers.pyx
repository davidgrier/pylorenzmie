from numpy.math cimport INFINITY
import cython
from scipy.optimize import OptimizeResult
import numpy as np
cimport numpy as np

np.import_array()

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def amoeba(objective,
           np.ndarray[DTYPE_t, ndim=1] x0,
           np.ndarray[DTYPE_t, ndim=1] xmin,
           np.ndarray[DTYPE_t, ndim=1] xmax,
           np.ndarray[DTYPE_t, ndim=1] simplex_scale,
           np.ndarray[DTYPE_t, ndim=1] xtol,
           initial_simplex=None, maxevals=int(1e3), ftol=1e-7):
    '''Nelder-mead optimization adapted from scipy.optimize.fmin'''
    # Initialize simplex
    cdef np.ulong_t N = x0.shape[0]
    cdef DTYPE_t Nf = np.float64(N)
    cdef np.ulong_t i, j
    cdef np.ndarray[DTYPE_t, ndim= 2] simplex = np.empty([N+1, N])
    cdef np.ndarray[DTYPE_t, ndim= 2] temp2d = np.empty([N+1, N])
    cdef np.ndarray[DTYPE_t, ndim= 1] temp = np.empty(N)
    simplex[0] = x0
    for i in range(N):
        simplex[i+1] = x0
        simplex[i+1, i] += simplex_scale[i]
    # Initialize algorithm
    cdef int max_nfev = maxevals
    cdef DTYPE_t fcntol = ftol
    cdef int neval = 1
    cdef int niter = 1
    cdef np.ndarray[DTYPE_t, ndim= 1] evals = np.empty(N+1)
    cdef np.ndarray[np.int_t, ndim= 1] idxs = np.empty(N+1,
                                                        dtype=np.int)
    for i in range(N+1):
        for j in range(N):
            temp[j] = simplex[i, j]
        minimum(temp, xmax)
        maximum(temp, xmin)
        for j in range(N):
            simplex[i, j] = temp[j]
        evals[i] = objective(temp)
        neval += 1
    sort(evals, idxs)
    take(simplex, idxs, temp2d)

    cdef DTYPE_t rho = 1.
    cdef DTYPE_t chi = 2.
    cdef DTYPE_t psi = 0.5
    cdef DTYPE_t sigma = 0.5

    cdef np.ndarray[DTYPE_t, ndim= 1] xbar = np.empty(N)
    cdef np.ndarray[DTYPE_t, ndim= 1] xr = np.empty(N)
    cdef np.ndarray[DTYPE_t, ndim= 1] xe = np.empty(N)
    cdef np.ndarray[DTYPE_t, ndim= 1] xc = np.empty(N)
    cdef np.ndarray[DTYPE_t, ndim= 1] xcc = np.empty(N)
    cdef DTYPE_t fxr, fxe, fxc, fxcc

    # START FITTING
    cdef str message = 'failure (hit max evals)'
    cdef bint end, check
    cdef DTYPE_t tempf, diff
    cdef int doshrink

    while(neval < max_nfev):
        # Test if simplex is small
        end = True
        for i in range(N):
            tempf = simplex[0, i]
            for j in range(1, N+1):
                diff = simplex[j, i] - tempf
                if diff < 0.:
                    diff = diff * -1.
                if diff > xtol[i]:
                    end = False
                    break
            if end is False:
                break
        if end:
            message = 'convergence (simplex small)'
            break
        # Test if function values are similar
        tempf = -INFINITY
        for i in range(1, N+1):
            diff = evals[0] - evals[i]
            if diff < 0.:
                diff = diff * -1.
            if diff > tempf:
                tempf = diff
        if tempf <= fcntol:
            message = 'convergence (fvals similar)'
            break
        # Test if simplex hits edge of parameter space
        end = False
        for j in range(N):
            for i in range(N+1):
                if xmax[j] == simplex[i, j] or xmin[j] == simplex[i, j]:
                    end = True
        if end:
            message = 'failure (stuck to boundary)'
            break
        # Reflection
        for j in range(N):
            tempf = 0.
            for i in range(N):
                tempf += simplex[i, j]
            xbar[j] = tempf / Nf
        for j in range(N):
            xr[j] = (1 + rho) * xbar[j] - rho * simplex[N, j]
        minimum(xr, xmax)
        maximum(xr, xmin)
        fxr = objective(xr)
        neval += 1
        doshrink = 0
        # Check if reflection is better than best estimate
        if fxr < evals[0]:
            # If so, reflect double and see if that's even better
            for j in range(N):
                xe[j] = (1. + rho * chi) * xbar[j]\
                    - rho * chi * simplex[N, j]
            minimum(xe, xmax)
            maximum(xe, xmin)
            fxe = objective(xe)
            neval += 1
            if fxe < fxr:
                for j in range(N):
                    simplex[N, j] = xe[j]
                evals[N] = fxe
            else:
                for j in range(N):
                    simplex[N, j] = xr[j]
                evals[N] = fxr
        else:
            if fxr < evals[N-1]:
                for j in range(N):
                    simplex[N, j] = xr[j]
                evals[N] = fxr
            else:
                # If reflection is not better, contract.
                if fxr < evals[N]:
                    for j in range(N):
                        xc[j] = (1. + psi * rho) * xbar[j] \
                            - psi * rho * simplex[N, j]
                    minimum(xc, xmax)
                    maximum(xc, xmin)
                    fxc = objective(xc)
                    neval += 1
                    if fxc <= fxr:
                        for j in range(N):
                            simplex[N, j] = xc[j]
                        evals[N] = fxc
                    else:
                        doshrink = 1
                else:
                    # Do 'inside' contraction
                    for j in range(N):
                        xcc[j] = (1 - psi) * xbar[j] \
                            + psi * simplex[N, j]
                    minimum(xcc, xmax)
                    maximum(xcc, xmin)
                    fxcc = objective(xcc)
                    neval += 1
                    if fxcc < evals[N]:
                        for j in range(N):
                            simplex[N, j] = xcc[j]
                        evals[N] = fxcc
                    else:
                        doshrink = 1
                if doshrink:
                    for i in range(1, N+1):
                        for j in range(N):
                            simplex[i, j] = simplex[0, j] + sigma * \
                                (simplex[i, j] - simplex[0, j])
                            temp[j] = simplex[i, j]
                        minimum(temp, xmax)
                        maximum(temp, xmin)
                        for j in range(N):
                            simplex[i, j] = temp[j]
                        evals[i] = objective(temp)
                        neval += 1
        sort(evals, idxs)
        take(simplex, idxs, temp2d)
        niter += 1
    success = False if 'failure' in message else True
    return OptimizeResult(x=simplex[0],
                          success=success,
                          message=message,
                          nit=niter,
                          nfev=neval,
                          fun=evals[0])


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void take(np.ndarray[DTYPE_t, ndim=2] a,
               np.ndarray[np.int_t, ndim=1] idxs,
               np.ndarray[DTYPE_t, ndim=2] b):
    cdef np.ulong_t nx, ny
    nx = a.shape[0]
    ny = a.shape[1]
    cdef np.ulong_t i
    for i in range(nx):
        for j in range(ny):
            b[i, j] = a[idxs[i], j]
    for i in range(nx):
        for j in range(ny):
            a[i, j] = b[i, j]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void minimum(np.ndarray[DTYPE_t, ndim=1] buff,
                  np.ndarray[DTYPE_t, ndim=1] bounds):
    cdef np.ulong_t n = buff.shape[0]
    cdef DTYPE_t m
    cdef np.ulong_t i
    for i in range(n):
        if buff[i] > bounds[i]:
            buff[i] = bounds[i]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void maximum(np.ndarray[DTYPE_t, ndim=1] buff,
                  np.ndarray[DTYPE_t, ndim=1] bounds):
    cdef np.ulong_t n = buff.shape[0]
    cdef DTYPE_t m
    cdef np.ulong_t i
    for i in range(n):
        if buff[i] < bounds[i]:
            buff[i] = bounds[i]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void sort(np.ndarray[DTYPE_t, ndim=1] a,
               np.ndarray[np.int_t, ndim=1] order):
    cdef np.ulong_t m = a.shape[0]
    cdef np.ulong_t i, j
    cdef DTYPE_t key

    for i in range(m):
        order[i] = i

    for i in range(1, m):
        key = a[i]
        j = i-1
        while j >= 0 and key < a[j]:
            a[j+1] = a[j]
            order[j+1] = order[j]
            j -= 1
        a[j+1] = key
        order[j+1] = i
