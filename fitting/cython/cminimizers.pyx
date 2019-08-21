from scipy.optimize import OptimizeResult
import numpy as np
import cython


@cython.boundscheck(False)
@cython.wraparound(False)
def amoeba(objective, x0, xmin, xmax, maxevals=int(1e3), initial_simplex=None,
           simplex_scale=.1, xtol=1e-7, ftol=1e-7, adaptive=False):
    '''Nelder-mead optimization adapted from scipy.optimize.fmin'''
    simplex_scale = np.asarray(simplex_scale)
    xtol = np.asarray(xtol)
    # Initialize simplex
    N = len(x0)
    if initial_simplex is None:
        if simplex_scale.size == 1:
            simplex_scale = np.full(N, simplex_scale[0])
        simplex = np.vstack([x0, np.diag(simplex_scale) + x0])
    else:
        if initial_simplex.shape != (N+1, N):
            raise ValueError("Initial simplex must be dimension (N+1, N)")
        simplex = initial_simplex
    # Initialize algorithm
    maxevals = maxevals
    neval = 1
    niter = 1
    one2np1 = list(range(1, N + 1))
    evals = np.zeros(N+1, float)
    for idx in range(N+1):
        simplex[idx] = np.maximum(xmin, np.minimum(simplex[idx], xmax))
        evals[idx] = objective(simplex[idx])
        neval += 1
    idxs = np.argsort(evals)
    evals = np.take(evals, idxs, 0)
    simplex = np.take(simplex, idxs, 0)

    if adaptive:
        dim = float(len(x0))
        rho = 1
        chi = 1 + 2/dim
        psi = 0.75 - 1/(2*dim)
        sigma = 1 - 1/dim
    else:
        rho = 1
        chi = 2
        psi = 0.5
        sigma = 0.5

    # START FITTING
    message = 'failure (hit max evals)'
    while(neval < maxevals):
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
