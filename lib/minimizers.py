import numpy as np
from scipy.optimize import OptimizeResult


def amoeba(objective, x0, xmin, xmax,
           simplex_scale=.1, xtol=1e-7, ftol=1e-7,
           maxevals=int(1e3), initial_simplex=None):
    '''
    Nelder-Mead optimization adapted from scipy.optimize.fmin

    Arguments
    ---------
    objective : callable
        Scalar objective function to be minimized
    x0 : np.ndarray
        [N] Initial guess for solution in space of N parameters
    xmin : np.ndarray
        [N] Lower bounds for parameters. These should be 
        far lower than the values the simplex explores
        and is only meant to catch the simplex if it runs
        far off from the solution
    xmax : np.ndarray
        [N] Upper bounds for parameters. See xmin documentation
        for usage

    Keywords
    --------
    simplex_scale : np.ndarray or float
        [N] Scale factor for each parameter in generating an 
        initial simplex.
    xtol : np.ndarray or float
        [N] Tolerance in each parameter for convergence. The
        algorithm stops when all values in the simplex are 
        within xtol of each other
    ftol : float
        Tolerance in objective function for convergence. The
        algorithm stops when all function values in simplex are 
        within ftol of each other.
    maxevals : int
        Max number of function evaluations before function quits
    initial_simplex : np.ndarray
        [N+1, N] Initial simplex. If None, simplex_scale is used to
        generate an initial simplex

    Returns
    -------
    result : scipy.optimize.OptimizeResult
        Contains information about solution, best function value,
        number of function evaluations and iterations, reason for
        termination, and success of the fit. See scipy's documentation.
    '''
    simplex_scale = np.asarray(simplex_scale)
    xtol = np.asarray(xtol)
    # Initialize simplex
    N = len(x0)
    if initial_simplex is None:
        if type(simplex_scale) is float:
            simplex_scale = np.full(N, simplex_scale)
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
        if np.max(np.abs(evals[0] - evals[1:])) <= ftol:
            message = 'convergence (fvals similar)'
            break
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
        xbar = np.add.reduce(simplex[:-1], 0) / N
        xr = (1 + rho) * xbar - rho * simplex[-1]
        xr = np.maximum(xmin, np.minimum(xr, xmax))
        fxr = objective(xr)
        neval += 1
        doshrink = 0
        # Check if reflection is better than best estimate
        if fxr < evals[0]:
            # If so, reflect double and see if that's even better
            xe = (1 + rho * chi) * xbar - rho * chi * simplex[-1]
            xe = np.maximum(xmin, np.minimum(xe, xmax))
            fxe = objective(xe)
            neval += 1
            if fxe < fxr:
                simplex[-1] = xe
                evals[-1] = fxe
            else:
                simplex[-1] = xr
                evals[-1] = fxr
        else:
            if fxr < evals[-2]:
                simplex[-1] = xr
                evals[-1] = fxr
            else:
                # If reflection is not better, contract.
                if fxr < evals[-1]:
                    xc = (1 + psi * rho) * xbar - psi * rho * simplex[-1]
                    xc = np.maximum(xmin, np.minimum(xc, xmax))
                    fxc = objective(xc)
                    neval += 1
                    if fxc <= fxr:
                        simplex[-1] = xc
                        evals[-1] = fxc
                    else:
                        doshrink = 1
                else:
                    # Do 'inside' contraction
                    xcc = (1 - psi) * xbar + psi * simplex[-1]
                    xcc = np.maximum(xmin, np.minimum(xcc, xmax))
                    fxcc = objective(xcc)
                    neval += 1
                    if fxcc < evals[-1]:
                        simplex[-1] = xcc
                        evals[-1] = fxcc
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
    result = OptimizeResult(x=best, success=success, message=message,
                            nit=niter, nfev=neval, fun=chi)
    return result
