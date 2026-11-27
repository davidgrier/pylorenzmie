'''Levenberg-Marquardt with Broyden updates of the Jacobian.

A Jacobian by finite differences costs one evaluation of the model for each
parameter, so that most of the time of a fit of a hologram goes into
Jacobians.  :func:`broyden_least_squares` computes one at the start, and then
corrects it after each step with what the step showed (Broyden's update).
It computes it again only when a step does worse than expected, and to
confirm that the fit has converged.

The fit stops when the step of Gauss-Newton predicts a fall of chi-squared,
in units of the reduced chi-squared, below a tolerance ``tol``.  Unlike a
tolerance on the relative change of the sum of squares, this does not depend
on the number of pixels.  If the model explains the data, and the prediction
is good, the fit is then within about ``sqrt(tol)`` of its uncertainties of
the minimum.  When the residuals are large, because the model does not
explain the data, the prediction can be far too small, and a fit stops much
farther from the minimum than that: choose a small ``tol``.

A model in single precision limits how well a difference gives the Jacobian,
and so how small the predicted fall can be.  The fit therefore also stops
when a step from a newly computed Jacobian fails although it was predicted to
gain less than the tolerance, and changed chi-squared, in the same units, by
less than one.

:class:`~pylorenzmie.analysis.Optimizer.Optimizer` uses it when the
``method`` of its settings is ``'broyden'``.

References
----------
1. C. G. Broyden, "A class of methods for solving nonlinear simultaneous
   equations," *Mathematics of Computation* **19**, 577-593 (1965).
2. H. B. Nielsen, "Damping parameter in Marquardt's method," Technical Report
   IMM-REP-1999-05, Technical University of Denmark (1999).
'''

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult

Vector = NDArray[np.floating]
Matrix = NDArray[np.floating]
Residuals = NDArray[Any]      # of whatever precision the model computes in

_MESSAGES = {
    0: 'The number of evaluations reached max_nfev.',
    1: 'The predicted fall of chi-squared is below the tolerance.',
    2: 'The step is too small to change the parameters.',
    3: 'No step lowers the sum of squares.',
    4: 'No step changes the sum of squares by more than the tolerance.'}

#: Damping at which no step is left to try.
_MAX_DAMPING = 1e14


def broyden_least_squares(fun: Callable[[Vector], Residuals],
                          x0: Vector,
                          jac: Callable[[Vector, Residuals], Matrix],
                          tol: float = 1e-4,
                          max_nfev: int | None = None) -> OptimizeResult:
    '''Minimize a sum of squares with few evaluations of the Jacobian.

    Parameters
    ----------
    fun : callable
        ``fun(x)`` returns the residuals at ``x``, in units of their noise,
        shape ``(m,)``.  Residuals of a point that must not be accepted are
        large or not finite; those at ``x0`` must be finite.
    x0 : numpy.ndarray
        Starting values, shape ``(n,)``.
    jac : callable
        ``jac(x, residuals)`` returns the Jacobian of ``fun`` at ``x``, shape
        ``(m, n)``; ``residuals`` is ``fun(x)``, for a finite difference.
    tol : float, optional
        The fit stops when the predicted fall of the sum of squares, divided
        by the sum of squares per degree of freedom, is below this.
        Default: 1e-4 (see the notes of the module).
    max_nfev : int, optional
        Largest number of calls of ``fun``, not counting those of ``jac``.
        Default: 100 for each parameter.

    Returns
    -------
    result : scipy.optimize.OptimizeResult
        With ``x``; ``cost``, half of the sum of squares; ``fun``, the
        residuals; ``jac``, the Jacobian computed (not updated) at ``x``;
        ``nfev`` and ``njev``; ``status`` (0 the budget was spent, 1
        converged, 2 the step vanished, 3 no step lowers the sum, 4 the
        sum is at the resolution of the model), ``success`` (status 1, 2
        or 4) and ``message``.

    Raises
    ------
    ValueError
        If ``tol`` is not positive.

    Notes
    -----
    The parameters are scaled by the largest norm that their column of the
    Jacobian has had, as MINPACK does.
    '''
    if not tol > 0.:
        raise ValueError(f'tol must be positive, not {tol}')
    x = np.array(x0, dtype=float)
    if max_nfev is None:
        max_nfev = 100 * x.size
    residuals = fun(x)
    cost = float(residuals @ residuals)
    freedom = max(residuals.size - x.size, 1)
    jacobian = jac(x, residuals)
    fresh = True            # computed at x, not updated since
    nfev, njev = 1, 1
    scale = np.zeros(x.size)
    damping, growth = 1e-3, 2.

    while True:
        scale = np.maximum(scale, np.linalg.norm(jacobian, axis=0))
        weight = np.where(scale > 0., scale, 1.)
        scaled = jacobian / weight
        values, vectors = np.linalg.eigh(scaled.T @ scaled)
        gradient = vectors.T @ (scaled.T @ residuals)
        # the fall of the sum of squares that Gauss-Newton predicts
        keep = values > 1e-14 * values[-1]
        predicted = np.sum(gradient[keep]**2 / values[keep])
        if predicted <= tol * cost / freedom:
            if fresh:
                status = 1
                break
            # an updated Jacobian is not trusted to say that we are done
            jacobian, fresh = jac(x, residuals), True
            njev += 1
            continue
        if nfev >= max_nfev:
            status = 0
            break
        direction = -vectors @ (gradient
                                / (np.maximum(values, 0.) + damping))
        step = direction / weight
        trial = fun(x + step)
        nfev += 1
        trial_cost = float(trial @ trial)
        linear = residuals + jacobian @ step
        expected = cost - float(linear @ linear)
        ratio = (cost - trial_cost) / expected if expected > 0. else -1.
        was_fresh, computed = fresh, jacobian
        # A trial far worse than the start (a point that the model does not
        # compute, say) says nothing about the slope here
        weighted = weight**2 * step
        length = float(step @ weighted)
        if trial_cost < 100. * cost and length > 0.:
            jacobian = jacobian + np.outer(trial - linear, weighted / length)
            fresh = False
        if trial_cost < cost:
            vanished = (np.linalg.norm(direction)
                        <= 1e-10 * np.linalg.norm(weight * x))
            x, residuals, cost = x + step, trial, trial_cost
            fresh = False
            damping *= max(1. / 3., 1. - (2. * ratio - 1.)**3)
            growth = 2.
            if vanished:
                status = 2
                break
            if ratio < 0.25:
                jacobian, fresh = jac(x, residuals), True
                njev += 1
        else:
            unit = cost / freedom       # one unit of reduced chi-squared
            # (a step too small to move x tries nothing)
            if (was_fresh and expected <= tol * unit
                    and trial_cost - cost <= unit
                    and np.any(x + step != x)):
                # the new Jacobian sees nothing left to gain, and the
                # trial that failed no change that matters: this is the
                # minimum, as well as the model knows it
                jacobian, fresh = computed, True
                status = 4
                break
            damping *= growth
            growth *= 2.
            if damping > _MAX_DAMPING:
                status = 3
                break
            # a failed step corrects a new Jacobian; an updated one that
            # failed is computed again
            if not was_fresh:
                jacobian, fresh = jac(x, residuals), True
                njev += 1

    if not fresh:           # the uncertainties need the slope at x
        jacobian = jac(x, residuals)
        njev += 1
    return OptimizeResult(x=x,
                          cost=0.5 * cost,
                          fun=residuals,
                          jac=jacobian,
                          nfev=nfev,
                          njev=njev,
                          status=status,
                          success=status in (1, 2, 4),
                          message=_MESSAGES[status])
