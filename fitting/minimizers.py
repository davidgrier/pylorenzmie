import numpy as np
import multiprocessing as mp
from lmfit.minimizer import MinimizerResult


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

    '''
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

    '''
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


def amoeba(objective, params, bounds, maxevals=int(1e3), initial_simplex=None,
           simplex_scale=.1, xtol=1e-7, ftol=1e-7, adaptive=False):
    '''Nelder-mead optimization adapted from scipy.optimize.fmin'''
    parameters = list(params.keys())
    if type(simplex_scale) == list:
        simplex_scale = np.array(simplex_scale)
    # Raise exception if any params are unbounded
    for param in parameters:
        if params[param].vary:
            #min, max = (params[param].min, params[param].max)
            min, max = (bounds[param][0], bounds[param][1])
            if not np.isfinite(min) or not np.isfinite(max):
                msg = "Nelder-mead requires finite parameter bounds"
                raise ValueError(msg)
    # If xtol is a dict then convert to array
    temp = []
    if type(xtol) == dict:
        for param in parameters:
            if params[param].vary:
                temp.append(xtol[param])
    xtol = np.array(temp)
    temp = []
    if type(simplex_scale) == dict:
        for param in parameters:
            if params[param].vary:
                temp.append(simplex_scale[param])
    simplex_scale = np.array(temp)
    # Initialize first guess normalization
    x0, scale, offset = ([], [], [])
    init_vals = {}
    for param in parameters:
        init_vals[param] = params[param].value
        if params[param].vary:
            x0.append(params[param].value)
            scale.append(params[param].max -
                         params[param].min)
            offset.append(params[param].min)
    x0, scale, offset = (np.array(x0), np.array(scale), np.array(offset))
    scale = None
    # Normalize
    # if type(simplex_scale) is np.ndarray:
    #    simplex_scale /= scale
    # if type(xtol) is np.ndarray:
    #    xtol /= scale
    # x0 = (x0 - offset) / scale
    # Initialize simplex
    N = len(x0)
    if initial_simplex is None:
        if type(simplex_scale) is float:
            simplex_scale = np.full(N, simplex_scale)
        print(simplex_scale)
        simplex = np.vstack([x0, np.diag(simplex_scale) + x0])
        # Make initial guess centroid of simplex
        xbar = np.add.reduce(simplex[:-1], 0) / N
        # simplex = simplex - (xbar - x0)
    else:
        if initial_simplex.shape != (N+1, N):
            raise ValueError("Initial simplex must be dimension (N+1, N)")
        # simplex = (initial_simplex - offset) / scale
        simplex = initial_simplex
    # Initialize algorithm
    N = len(x0)
    maxevals = maxevals
    penalty = 1000
    neval = 1
    niter = 1
    one2np1 = list(range(1, N + 1))
    evals = np.zeros(N+1, float)
    for idx in range(N+1):
        params, penalize = _updateParams(simplex[idx], params,
                                         bounds, scale, offset)
        evals[idx] = objective(params)
        if penalize:
            evals[idx] *= penalty
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
        params, penalize = _updateParams(xr, params, bounds,
                                         scale, offset)
        fxr = objective(params)
        if penalize:
            fxr *= penalty
        neval += 1
        doshrink = 0
        # Check if reflection is better than best estimate
        if fxr < evals[0]:
            # If so, reflect double and see if that's even better
            xe = (1 + rho * chi) * xbar - rho * chi * simplex[-1]
            params, penalize = _updateParams(xe, params, bounds,
                                             scale, offset)
            fxe = objective(params)
            if penalize:
                fxe *= penalty
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
                    params, penalize = _updateParams(xc, params, bounds,
                                                     scale, offset)
                    fxc = objective(params)
                    if penalize:
                        fxc *= penalty
                    neval += 1
                    if fxc <= fxr:
                        simplex[-1] = xc
                        evals[-1] = fxc
                    else:
                        doshrink = 1
                else:
                    # Do 'inside' contraction
                    xcc = (1 - psi) * xbar + psi * simplex[-1]
                    params, penalize = _updateParams(xcc, params, bounds,
                                                     scale, offset)
                    fxcc = objective(params)
                    if penalize:
                        fxcc *= penalty
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
                        params, penalize = _updateParams(simplex[j], params, bounds,
                                                         scale, offset)
                        evals[j] = objective(params)
                        if penalize:
                            evals[j] *= penalty
                        neval += 1
        idxs = np.argsort(evals)
        simplex = np.take(simplex, idxs, 0)
        evals = np.take(evals, idxs, 0)

        niter += 1
    params, penalize = _updateParams(simplex[0], params, bounds,
                                     scale, offset)
    chi = evals[0]
    success = False if neval == maxevals else True
    result = MinimizerResult(params=params,
                             nvarys=len(x0),
                             chisqr=chi,
                             nfev=neval, success=success,
                             init_vals=init_vals,
                             errorbars=False)
    result.method = 'Nelder-Mead (custom)'
    return result


def _updateParams(x, params, bounds, scale, offset):
    # print(x)
    # x = xn * scale + offset
    parameters = list(params.keys())
    varying = []
    for param in parameters:
        if params[param].vary:
            varying.append(param)
    penalize = False
    for idx in range(len(x)):
        params[varying[idx]].value = x[idx]
        if bounds[varying[idx]][1] < params[varying[idx]].value:
            penalize = True
        if bounds[varying[idx]][0] > params[varying[idx]].value:
            penalize = True
    print(penalize)
    return params, penalize
