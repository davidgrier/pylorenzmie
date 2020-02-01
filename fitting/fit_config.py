import numpy as np
import json

vary = [True] * 5
vary.extend([False] * 5)
# ... levenberg-marquardt variable scale factor
x_scale = [1.e4, 1.e4, 1.e3, 1.e4, 1.e5, 1.e7, 1.e2, 1.e2, 1.e2, 1]
# ... bounds around intial guess for bounded nelder-mead
simplex_bounds = [(-np.inf, np.inf), (-np.inf, np.inf),
                  (0., 2000.), (.05, 4.), (1., 3.),
                  (0., 3.), (1., 3.), (.100, 2.00), (0., 1.),
                  (0., 5.)]
# ... scale of initial simplex
simplex_scale = [4., 4., 5., 0.01, 0.01, .2, .1, .1, .05, .05]
# ... tolerance for nelder-mead termination
simplex_tol = [.1, .1, .01, .001, .001, .001, .01, .01, .01, .01]
# Default options for amoeba and lm not parameter dependent
lm_options = {'method': 'lm',
              'xtol': 1.e-6,
              'ftol': 1.e-3,
              'gtol': 1e-6,
              'max_nfev': 2000,
              'diff_step': 1e-5,
              'verbose': 0}
nm_options = {'ftol': 1.e-3, 'maxevals': 800}

globalparams = ("z_p", "a_p", "n_p")
# Gaussian well standard deviation
well_std = [None, None, 5, .03, .02, None, None, None, None, None]
# Sampling range for globalized optimization based on Estimator
sample_range = [None, None, 30, .2, .1, None, None, None, None, None]
sample_options = {"independent": True, "distribution": "wells"}

d = {}
d['vary'] = vary
d['nm'] = {}
d['nm']['simplex_bounds'] = simplex_bounds
d['nm']['simplex_scale'] = simplex_scale
d['nm']['simplex_tol'] = simplex_tol
d['nm']['options'] = nm_options
d['lm'] = {}
d['lm']['x_scale'] = x_scale
d['lm']['options'] = lm_options
d['global'] = {}
d['global']['params'] = globalparams
d['global']['options'] = sample_options
d['global']['well_std'] = well_std
d['global']['sample_range'] = sample_range

with open('.LMHologram', 'w') as f:
    json.dump(d, f)
