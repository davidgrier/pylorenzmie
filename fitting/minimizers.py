import numpy as np


def amoeba(objective, x0, maxevals=int(1e3), initial_simplex=None,
           simplex_scale=.1, xtol=1e-7, ftol=1e-7, adaptive=False):
    '''Nelder-mead optimization adapted from scipy.optimize.fmin'''
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
    while(neval < maxevals):
        if (all(np.amax(np.abs(simplex[1:] - simplex[0]), axis=0) <= xtol) and
                np.max(np.abs(simplex[0] - simplex[1:])) <= ftol):
            break
        # Reflect
        xbar = np.add.reduce(simplex[:-1], 0) / N
        xr = (1 + rho) * xbar - rho * simplex[-1]
        fxr = objective(xr)
        neval += 1
        doshrink = 0
        # Check if reflection is better than best estimate
        if fxr < evals[0]:
            # If so, reflect double and see if that's even better
            xe = (1 + rho * chi) * xbar - rho * chi * simplex[-1]
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
                        evals[j] = objective(simplex[j])
                        neval += 1
        idxs = np.argsort(evals)
        simplex = np.take(simplex, idxs, 0)
        evals = np.take(evals, idxs, 0)
        niter += 1

    best = simplex[0]
    chi = evals[0]
    success = False if neval == maxevals else True
    return best, chi, neval, niter, success


'''
def amoebas(objective, params, initial_simplex=None, maxevals=int(1e3),
            simplex_scale=.1, namoebas=2, xtol=1e-7, ftol=1e-7):
    parameters = list(params.keys())
    temp = []
    if type(simplex_scale) == dict:
        for param in parameters:
            if params[param].vary:
                temp.append(simplex_scale[param])
        simplex_scale = np.array(temp)
    x0 = []
    for param in params.keys():
        if params[param].vary:
            x0.append(params[param].value)
    x0 = np.array(x0)
    N = len(x0)
    if initial_simplex is None:
        if namoebas == 1:
            scales = [np.array(simplex_scale)]
        else:
            scales = np.linspace(-simplex_scale,
                                 simplex_scale,
                                 namoebas)
        initial_simplex = []
        for scale in scales:
            if type(scale) is np.float64:
                scale = np.full(N, scale)
            simplex = np.vstack([x0, np.diag(scale) + x0])
            # Make initial guess centroid of simplex
            xbar = np.add.reduce(simplex[:-1], 0) / N
            # simplex = simplex - (xbar - x0)
            initial_simplex.append(simplex)
    minresult = None
    minchi = np.inf

    
    mp.set_start_method('spawn')
    pool = mp.Pool(nsimp)
    args = [(objective, params,
             simplex, delta, xtol, ftol) for simplex in initial_simplex]
    results = pool.starmap(amoeba, args)
    pool.close()
    pool.terminate()
    pool.join()
    for result in results:
        if result.redchi < minchi:
            minresult = result
            minchi = result.redchi
        # report_fit(result)

    
    chis = []
    for idx, simplex in enumerate(initial_simplex):
        result = amoeba(objective, params,
                        initial_simplex=simplex,
                        xtol=xtol, ftol=ftol,
                        maxevals=maxevals)
        if result.chisqr < minchi:
            minresult = result
            minchi = result.chisqr
        chis.append(result.chisqr)
    minresult.chis = np.array(chis)
    return minresult
'''
