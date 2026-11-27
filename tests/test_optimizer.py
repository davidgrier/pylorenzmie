import importlib
import unittest
from pathlib import Path
from unittest.mock import patch

from pylorenzmie.utilities import imread
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from pylorenzmie.analysis import Optimizer
from pylorenzmie.analysis.Optimizer import PENALTY
from pylorenzmie.analysis.Hologram import Hologram
from pylorenzmie.analysis.Mask import Mask
from pylorenzmie.lib import LMObject
from pylorenzmie.theory import LorenzMie
from pylorenzmie.theory.Sphere import MAX_RADIUS, ModelDomainError
try:
    from pylorenzmie.theory.jaxLorenzMie import jaxLorenzMie
    _jax_available = True
except Exception:
    _jax_available = False

# the package namespace holds the class of the same name
Sphere_module = importlib.import_module('pylorenzmie.theory.Sphere')

THIS_DIR = Path(__file__).parent.resolve()
TEST_IMAGE = str(THIS_DIR / 'data' / 'crop.png')


class TestOptimizer(unittest.TestCase):

    def setUp(self):
        img = imread(TEST_IMAGE).astype(float)
        img /= np.mean(img)
        shape = img.shape
        coords = LMObject.meshgrid(shape)
        model = LorenzMie(coordinates=coords)
        model.instrument.wavelength = 0.447
        model.instrument.magnification = 0.048
        model.instrument.n_m = 1.34
        model.particle.r_p = [shape[0] // 2, shape[1] // 2, 330]
        model.particle.a_p = 1.1
        model.particle.n_p = 1.4
        self.hologram = Hologram(img)
        self.optimizer = Optimizer(model=model)

    def test_result_none(self):
        '''result is None before any optimization has run.'''
        self.assertIs(self.optimizer.result, None)

    def test_mask_default_none(self):
        '''mask defaults to None (use all pixels).'''
        self.assertIsNone(self.optimizer.mask)

    def test_mask_set(self):
        '''mask can be set to a Mask instance.'''
        mask = Mask(fraction=0.1)
        opt = Optimizer(model=self.optimizer.model, mask=mask)
        self.assertIs(opt.mask, mask)

    def test_metadata(self):
        self.assertIsInstance(self.optimizer.metadata, pd.Series)

    def test_properties(self):
        properties = self.optimizer.properties
        self.optimizer.properties = properties
        self.assertTrue('settings' in properties)

    def test_variables_setter_updates_fixed(self):
        '''Setting variables derives fixed from model.properties.'''
        variables = ['x_p', 'y_p', 'z_p', 'a_p']
        self.optimizer.variables = variables
        for v in variables:
            self.assertIn(v, self.optimizer.variables)
            self.assertNotIn(v, self.optimizer.fixed)
        for f in self.optimizer.fixed:
            self.assertNotIn(f, variables)

    def test_fixed_setter_updates_variables(self):
        '''Setting fixed derives variables from model.properties.'''
        all_props = list(self.optimizer.model.properties)
        fixed = all_props[:2]
        self.optimizer.fixed = fixed
        for f in fixed:
            self.assertIn(f, self.optimizer.fixed)
            self.assertNotIn(f, self.optimizer.variables)

    def test_default_variables_are_the_particle_parameters(self):
        self.assertEqual(self.optimizer.variables,
                         ['x_p', 'y_p', 'z_p', 'a_p', 'n_p'])

    def test_aberrated_model_keeps_its_aberrations_fixed_by_default(self):
        from pylorenzmie.theory import Aberrated
        model = Aberrated(LorenzMie)(spherical=1.)
        optimizer = Optimizer(model=model)
        self.assertEqual(optimizer.variables,
                         ['x_p', 'y_p', 'z_p', 'a_p', 'n_p'])
        for name in ('defocus', 'spherical', 'coma', 'astigmatism',
                     'distortion', 'x_c', 'y_c'):
            self.assertIn(name, optimizer.fixed)

    def test_aberrations_can_still_be_freed(self):
        from pylorenzmie.theory import Aberrated
        optimizer = Optimizer(model=Aberrated(LorenzMie)())
        optimizer.variables = ['x_p', 'y_p', 'z_p', 'a_p', 'n_p', 'spherical']
        self.assertIn('spherical', optimizer.variables)
        self.assertNotIn('spherical', optimizer.fixed)

    def test_user_settings_are_kept(self):
        settings = {'method': 'dogbox', 'loss': 'linear', 'max_nfev': 7}
        optimizer = Optimizer(settings=settings)
        self.assertEqual(optimizer.settings['method'], 'dogbox')
        self.assertEqual(optimizer.settings['max_nfev'], 7)
        self.assertFalse(optimizer.robust)
        settings['max_nfev'] = 99               # the dictionary was copied
        self.assertEqual(optimizer.settings['max_nfev'], 7)

    def test_robust_true_overrides_settings(self):
        optimizer = Optimizer(settings={'method': 'lm', 'loss': 'linear'},
                              robust=True)
        self.assertEqual(optimizer.settings['loss'], 'cauchy')
        self.assertEqual(optimizer.settings['method'], 'trf')
        self.assertTrue(optimizer.robust)

    def test_robust_false_overrides_settings(self):
        optimizer = Optimizer(settings={'method': 'trf', 'loss': 'cauchy'},
                              robust=False)
        self.assertEqual(optimizer.settings['loss'], 'linear')
        self.assertEqual(optimizer.settings['method'], 'lm')

    def test_robust_matches_the_scale(self):
        self.optimizer.robust = True
        self.assertEqual(self.optimizer.settings['f_scale'], 'auto')
        self.optimizer.robust = False
        self.assertNotIn('f_scale', self.optimizer.settings)

    def test_robust_keeps_a_scale_that_was_given(self):
        optimizer = Optimizer(settings={'f_scale': 2.}, robust=True)
        self.assertEqual(optimizer.settings['f_scale'], 2.)
        optimizer.robust = False
        self.assertEqual(optimizer.settings['f_scale'], 2.)

    def test_a_loss_of_the_settings_keeps_the_scale_of_scipy(self):
        self.optimizer.settings.update(method='trf', loss='cauchy')
        self.assertNotIn('f_scale', self.optimizer.settings)

    def test_default_fixed_includes_noise(self):
        '''noise is fixed by default to prevent degenerate fits.'''
        self.assertIn('noise', self.optimizer.fixed)

    def test_fixed_list_is_copied(self):
        '''Mutating the list passed to fixed does not affect the optimizer.'''
        fixed = ['wavelength']
        opt = Optimizer(model=self.optimizer.model, fixed=fixed)
        fixed.append('n_m')
        self.assertNotIn('n_m', opt.fixed)

    def test_robust_false_uses_lm(self):
        self.optimizer.robust = False
        self.assertFalse(self.optimizer.robust)
        self.assertEqual(self.optimizer.settings['method'], 'lm')

    def test_robust_true_uses_trf(self):
        self.optimizer.robust = True
        self.assertTrue(self.optimizer.robust)
        self.assertEqual(self.optimizer.settings['method'], 'trf')

    def test_report_lists_every_fitted_parameter(self):
        '''report() after a fit names each parameter with its error.'''
        self.optimizer.settings['max_nfev'] = 2
        self.optimizer.optimize(self.hologram)
        report = self.optimizer.report()
        for name in self.optimizer.variables:
            self.assertIn(f'{name} = ', report)
        self.assertIn('±', report)
        self.assertIn('pixels', report)
        self.assertIn('npixels = ', report)
        self.assertIn('χ² = ', report)

    def test_report_before_optimize_raises(self):
        '''report() raises RuntimeError when called before optimize().'''
        with self.assertRaises(RuntimeError):
            self.optimizer.report()

    def test_optimize_returns_series(self):
        '''optimize() returns a pandas.Series.'''
        self.optimizer.settings['max_nfev'] = 1
        result = self.optimizer.optimize(self.hologram)
        self.assertIsInstance(result, pd.Series)

    def test_optimize_with_mask_returns_series(self):
        '''optimize() with a mask returns a pandas.Series.'''
        self.optimizer.mask = Mask(fraction=0.1)
        self.optimizer.settings['max_nfev'] = 1
        result = self.optimizer.optimize(self.hologram)
        self.assertIsInstance(result, pd.Series)

    def test_seeded_mask_makes_the_fit_reproducible(self):
        '''The same seed gives the same pixels and so the same result.'''
        self.optimizer.mask = Mask(fraction=0.1, seed=4)
        self.optimizer.settings['max_nfev'] = 3
        start = dict(self.optimizer.model.particle.properties)
        first = self.optimizer.optimize(self.hologram)
        self.optimizer.model.particle.properties = start
        second = self.optimizer.optimize(self.hologram)
        self.assertEqual(first['npix'], second['npix'])
        pd.testing.assert_series_equal(first, second)

    def test_unseeded_mask_has_a_fixed_pixel_count(self):
        self.optimizer.mask = Mask(fraction=0.1)
        self.optimizer.settings['max_nfev'] = 1
        counts = {self.optimizer.optimize(self.hologram)['npix']
                  for _ in range(4)}
        self.assertEqual(len(counts), 1)

    def test_optimize_npix_all_pixels(self):
        '''Without a mask, npix equals the total pixel count.'''
        self.optimizer.settings['max_nfev'] = 1
        result = self.optimizer.optimize(self.hologram)
        ny, nx = self.hologram.shape
        self.assertEqual(result['npix'], ny * nx)

    def test_optimize_npix_masked(self):
        '''With a mask, npix is less than the total pixel count.'''
        self.optimizer.mask = Mask(fraction=0.1)
        self.optimizer.settings['max_nfev'] = 1
        result = self.optimizer.optimize(self.hologram)
        ny, nx = self.hologram.shape
        self.assertLess(result['npix'], ny * nx)


@unittest.skipUnless(_jax_available, 'JAX not installed')
class TestOptimizerAnalyticalJac(unittest.TestCase):
    '''Verify Optimizer uses the analytical Jacobian when available.'''

    def setUp(self):
        from unittest.mock import MagicMock
        img = imread(TEST_IMAGE).astype(float)
        img /= np.mean(img)
        shape = img.shape
        coords = LMObject.meshgrid(shape)
        model = jaxLorenzMie(coordinates=coords)
        model.instrument.wavelength = 0.447
        model.instrument.magnification = 0.048
        model.instrument.n_m = 1.34
        model.particle.r_p = [shape[0] // 2, shape[1] // 2, 330]
        model.particle.a_p = 1.1
        model.particle.n_p = 1.4
        self.hologram = Hologram(img)
        self.optimizer = Optimizer(model=model)
        # Build a valid mock least_squares result for the default 5 variables
        n = len(self.optimizer.variables)
        m = img.size
        mock = MagicMock()
        mock.x = np.zeros(n)
        mock.cost = 1.0
        mock.success = True
        mock.jac = np.ones((m, n)) * 0.01  # rank-1, passes SVD in _statistics
        self.mock_result = mock
        self.n_variables = n
        self.n_pixels = m

    def _mock_result_for(self, n_vars):
        from unittest.mock import MagicMock
        mock = MagicMock()
        mock.x = np.zeros(n_vars)
        mock.cost = 1.0
        mock.success = True
        mock.jac = np.ones((self.n_pixels, n_vars)) * 0.01
        return mock

    def test_analytical_jac_selected(self):
        '''Optimizer passes analytical jac callable when all variables covered.'''
        from unittest.mock import patch
        with patch('pylorenzmie.analysis.Optimizer.least_squares',
                   return_value=self.mock_result) as mock_ls:
            self.optimizer.optimize(self.hologram)
        _, kwargs = mock_ls.call_args
        self.assertTrue(callable(kwargs.get('jac')),
                        'expected analytical jac callable in least_squares call')

    def test_analytical_jac_skipped_for_uncovered_variable(self):
        '''Optimizer omits jac when a variable lacks coverage (k_p).'''
        from unittest.mock import patch
        self.optimizer.variables = ['x_p', 'y_p', 'z_p', 'a_p', 'n_p', 'k_p']
        with patch('pylorenzmie.analysis.Optimizer.least_squares',
                   return_value=self._mock_result_for(6)) as mock_ls:
            self.optimizer.optimize(self.hologram)
        _, kwargs = mock_ls.call_args
        self.assertNotIn('jac',
                         kwargs,
                         'expected no jac kwarg when k_p not covered')

    def test_analytical_jac_shape(self):
        '''_analytical_jac returns (npts, n_variables) float64 array.'''
        self.optimizer._data = self.hologram.flat_data
        self.optimizer.model.coordinates = self.hologram.flat_coordinates
        values = np.array([self.optimizer.model.properties[p]
                           for p in self.optimizer.variables])
        J = self.optimizer._analytical_jac(values)
        self.assertEqual(J.shape, (self.n_pixels, self.n_variables))
        self.assertEqual(J.dtype, np.float64)

    def test_optimize_with_analytical_jac_returns_series(self):
        '''optimize() with analytical jac returns a pandas.Series.'''
        self.optimizer.settings['max_nfev'] = 1
        result = self.optimizer.optimize(self.hologram)
        self.assertIsInstance(result, pd.Series)


class TestOptimizerFit(unittest.TestCase):
    '''Fits of the image of one sphere, every fourth pixel.'''

    def setUp(self):
        img = imread(TEST_IMAGE).astype(float)
        img /= np.mean(img)
        img = img[::4, ::4]
        shape = img.shape
        coords = 4. * LMObject.meshgrid(shape)
        model = LorenzMie(coordinates=coords)
        model.instrument.wavelength = 0.447
        model.instrument.magnification = 0.048
        model.instrument.n_m = 1.34
        model.particle.r_p = [shape[0] // 2, shape[1] // 2, 330]
        model.particle.a_p = 1.1
        model.particle.n_p = 1.4
        self.img = img
        self.coords = coords
        self.hologram = Hologram.from_coordinates(img, coords.reshape(2, *shape))
        self.optimizer = Optimizer(model=model)

    def test_optimize(self):
        result = self.optimizer.optimize(self.hologram)
        self.assertTrue(result.success)

    def test_report_after_optimize(self):
        '''report() covers exactly the fitted variables after optimize()'''
        self.optimizer.optimize(self.hologram)
        report = self.optimizer.report()
        for v in self.optimizer.variables:
            self.assertIn(v, report)
        self.assertIn('χ²', report)

    def test_optimize_failure(self):
        shape = self.img.shape
        bad = Hologram.from_coordinates(self.img + 100.,
                                        self.coords.reshape(2, *shape))
        result = self.optimizer.optimize(bad)
        failure = not result.success or (result.redchi > 100.)
        self.assertTrue(failure)

    def test_bounds_keep_the_fit_within_limits(self):
        free = self.optimizer.optimize(self.hologram)
        a_free = free['a_p']
        self.optimizer.model.particle.a_p = 1.1
        limit = 1.1 + 0.5 * (a_free - 1.1)     # between start and optimum
        self.optimizer.bounds = {'a_p': (None, limit)
                                 if a_free > 1.1 else (limit, None)}
        result = self.optimizer.optimize(self.hologram)
        if a_free > 1.1:
            self.assertLessEqual(result['a_p'], limit + 1e-9)
        else:
            self.assertGreaterEqual(result['a_p'], limit - 1e-9)

    def test_bounds_select_trf_without_changing_the_settings(self):
        self.assertEqual(self.optimizer.settings['method'], 'lm')
        self.optimizer.bounds = {'a_p': (0.5, 2.)}
        result = self.optimizer.optimize(self.hologram)
        self.assertTrue(result.success)
        self.assertEqual(self.optimizer.settings['method'], 'lm')
        self.assertNotIn('bounds', self.optimizer.settings)

    def test_bounds_work_with_settings_that_have_no_method(self):
        self.optimizer.settings = {'ftol': 1e-6}
        self.optimizer.settings.pop('method', None)
        self.optimizer.bounds = {'a_p': (0.5, 2.)}
        result = self.optimizer.optimize(self.hologram)
        self.assertTrue(result.success)

    def test_a_start_outside_the_bounds_is_named(self):
        self.optimizer.bounds = {'a_p': (2., 3.)}     # the start is 1.1
        with self.assertRaisesRegex(ValueError, "'a_p'.*outside"):
            self.optimizer.optimize(self.hologram)

    def test_bounds_are_validated(self):
        with self.assertRaises(ValueError):
            self.optimizer.bounds = {'radius': (0., 1.)}
        with self.assertRaises(ValueError):
            self.optimizer.bounds = {'a_p': (2., 1.)}
        self.assertEqual(self.optimizer.bounds, {})

    def test_bounds_can_be_removed(self):
        self.optimizer.bounds = {'a_p': (0.5, 2.)}
        self.assertEqual(self.optimizer.bounds, {'a_p': (0.5, 2.)})
        self.optimizer.bounds = None
        self.assertEqual(self.optimizer.bounds, {})

    def test_a_step_beyond_the_largest_radius_is_rejected(self):
        # the model refuses a radius of more than MAX_RADIUS, so that an
        # unbounded fit cannot ask for all the memory of the computer, and
        # the fit rejects the step with a residual that it cannot accept
        self.optimizer.optimize(self.hologram)
        values = [self.optimizer.model.properties[p]
                  for p in self.optimizer.variables]
        values[self.optimizer.variables.index('a_p')] = 2. * MAX_RADIUS
        residuals = self.optimizer._residuals(np.array(values))
        self.assertEqual(residuals.shape, self.optimizer._pixels.shape)
        self.assertTrue(np.all(residuals == PENALTY))

    def test_a_starting_point_that_is_not_finite_is_named(self):
        shape = self.hologram.data.shape
        nan = np.full(shape[0] * shape[1], np.nan)
        model = self.optimizer.model
        with patch.object(model, 'hologram', return_value=nan):
            with self.assertRaisesRegex(ModelDomainError, 'not finite'):
                self.optimizer.optimize(self.hologram)

    def test_a_starting_point_beyond_the_largest_radius_is_refused(self):
        self.optimizer.model.particle.a_p = 2. * MAX_RADIUS
        with self.assertRaises(ModelDomainError):
            self.optimizer.optimize(self.hologram)

    def test_a_fit_that_steps_beyond_the_largest_radius_finishes(self):
        # the true radius is beyond the limit, which is lowered to 2 um
        # so that the model is fast: the fit wants to step past it
        shape = (31, 31)
        model = LorenzMie(coordinates=LMObject.meshgrid(shape))
        model.instrument.wavelength = 0.447
        model.instrument.magnification = 0.048
        model.instrument.n_m = 1.34
        model.particle.r_p = [15., 15., 200.]
        model.particle.a_p = 2.5
        model.particle.n_p = 1.5
        rng = np.random.default_rng(0)
        data = model.hologram().reshape(shape) + rng.normal(0., 0.05, shape)
        model.particle.a_p = 1.5
        with patch.object(Sphere_module, 'MAX_RADIUS', 2.):
            result = Optimizer(model=model).optimize(Hologram(data))
        self.assertIsInstance(result, pd.Series)
        self.assertLessEqual(abs(result['a_p']), 2.)

class TestOptimizerMatchedScale(unittest.TestCase):
    """f_scale='auto', for a simulated sphere with structured noise."""

    SHAPE = (51, 51)
    TRUTH = dict(x_p=25.3, y_p=24.6, z_p=150., a_p=0.75, n_p=1.45)

    def setUp(self):
        model = LorenzMie(coordinates=LMObject.meshgrid(self.SHAPE))
        model.properties = self.TRUTH
        clean = model.hologram().reshape(self.SHAPE)
        rng = np.random.default_rng(0)
        data = clean + rng.normal(0., 0.05, self.SHAPE)
        data[5:14, 30:39] += 0.8        # a defect that the model lacks
        self.clean = clean
        self.hologram = Hologram(data)
        self.model = model

    def fit(self, noise=0.05, hologram=None, **settings):
        start = dict(z_p=153., a_p=0.76, noise=noise)
        self.model.properties = self.TRUTH | start
        optimizer = Optimizer(model=self.model)
        optimizer.settings.update({'method': 'trf'} | settings)
        result = optimizer.optimize(hologram or self.hologram)
        self.assertEqual(optimizer.settings.get('f_scale'),
                         settings.get('f_scale'))
        return result

    def values(self, result):
        return result[list(self.TRUTH)].to_numpy(dtype=float)

    def test_is_two_fits_with_the_scale_of_the_first(self):
        auto = self.fit(loss='cauchy', f_scale='auto')
        linear = self.fit()
        residuals = (self.model.hologram()
                     - self.hologram.flat_data) / 0.05
        scale = 1.5 * 1.4826 * np.median(np.abs(residuals))
        optimizer = Optimizer(model=self.model)
        optimizer.settings.update(
            method='trf',
            loss='cauchy',
            f_scale=scale,
        )
        self.model.properties = {p: linear[p] for p in self.TRUTH}
        manual = optimizer.optimize(self.hologram)
        pd.testing.assert_series_equal(auto.drop('redchi'),
                                       manual.drop('redchi'))

    def test_redchi_is_that_of_a_scale_of_one(self):
        for loss in ('cauchy', 'huber', 'soft_l1', 'arctan'):
            with self.subTest(loss=loss):
                result = self.fit(loss=loss, f_scale='auto')
                residuals = (self.model.hologram()
                             - self.hologram.flat_data) / 0.05
                unit = least_squares(lambda _, r=residuals: r,
                                     [0.],
                                     loss=loss,
                                     max_nfev=1)
                ndeg = residuals.size - len(self.TRUTH)
                self.assertAlmostEqual(result['redchi'],
                                       2. * unit.cost / ndeg,
                                       places=8)

    def test_redchi_of_a_loss_that_is_a_function(self):
        def cauchy(z):
            return np.array([np.log1p(z), 1. / (1. + z), -1. / (1. + z)**2])

        named = self.fit(loss='cauchy', f_scale='auto')
        function = self.fit(loss=cauchy, f_scale='auto')
        pd.testing.assert_series_equal(function, named)

    def test_a_result_does_not_follow_later_settings(self):
        self.model.properties = dict(self.TRUTH, noise=0.05)
        optimizer = Optimizer(model=self.model, robust=True)
        result = optimizer.optimize(self.hologram)
        optimizer.robust = False
        pd.testing.assert_series_equal(optimizer.result, result)

    def test_does_not_depend_on_the_noise_of_the_instrument(self):
        right = self.fit(noise=0.05, loss='cauchy', f_scale='auto')
        small = self.fit(noise=0.01, loss='cauchy', f_scale='auto')
        np.testing.assert_allclose(self.values(small),
                                   self.values(right),
                                   rtol=1e-4)

    def test_discounts_a_defect(self):
        truth = np.array(list(self.TRUTH.values()))
        linear = np.abs(self.values(self.fit()) - truth)
        matched = self.fit(loss='cauchy', f_scale='auto')
        robust = np.abs(self.values(matched) - truth)
        self.assertLess(robust[2], linear[2])
        self.assertLess(robust[3], linear[3])

    def test_the_linear_loss_ignores_it(self):
        for method in ('lm', 'trf', 'broyden'):
            with self.subTest(method=method):
                pd.testing.assert_series_equal(
                    self.fit(method=method, f_scale='auto'),
                    self.fit(method=method))

    def test_keeps_the_bounds(self):
        self.model.properties = dict(self.TRUTH, a_p=0.76, noise=0.05)
        optimizer = Optimizer(model=self.model,
                              bounds={'a_p': (0.755, 0.8)})
        optimizer.settings.update(loss='cauchy', f_scale='auto')
        result = optimizer.optimize(self.hologram)
        self.assertGreaterEqual(result['a_p'], 0.755)

    def test_lm_cannot_do_a_robust_loss(self):
        with self.assertRaisesRegex(ValueError, "'lm'"):
            self.fit(method='lm', loss='cauchy', f_scale='auto')
        self.assertEqual(self.model.properties['a_p'], 0.76)

    def test_data_without_noise(self):
        clean = Hologram(self.clean)
        result = self.fit(hologram=clean, loss='cauchy', f_scale='auto')
        truth = list(self.TRUTH.values())
        np.testing.assert_allclose(self.values(result), truth, rtol=1e-5)


class _Stepped(LorenzMie):
    '''A calculator that gives the steps of its differences.'''

    diff_steps = {'x_p': 1e-4, 'y_p': 1e-4, 'z_p': 1e-3,
                  'a_p': 1e-6, 'n_p': 1e-6, 'k_p': 1e-6}


class TestOptimizerDifferences(unittest.TestCase):

    def setUp(self):
        self.truth = {'x_p': 40.3, 'y_p': 38.6, 'z_p': 150.,
                      'a_p': 0.8, 'n_p': 1.45}
        self.data = self.model(LorenzMie).hologram().reshape(80, 80)
        noise = np.random.default_rng(0).normal(0., 0.02, self.data.shape)
        self.hologram = Hologram(self.data + noise)
        self.start = {'x_p': 40., 'y_p': 39., 'z_p': 153.,
                      'a_p': 0.78, 'n_p': 1.46}

    def model(self, cls):
        model = cls(coordinates=cls.meshgrid((80, 80)))
        model.instrument.noise = 0.02
        model.properties = self.truth
        return model

    def optimizer(self, cls=_Stepped, **kwargs):
        optimizer = Optimizer(model=self.model(cls), **kwargs)
        optimizer.model.properties = self.start
        return optimizer

    def evaluated(self, optimizer):
        '''Ready to evaluate the residuals, without a fit.'''
        optimizer._data = self.hologram.flat_data
        return np.array([optimizer.model.properties[p]
                         for p in optimizer.variables], dtype=float)

    def test_the_model_steps_are_used(self):
        optimizer = self.optimizer()
        values = self.evaluated(optimizer)
        np.testing.assert_array_equal(optimizer._steps(values),
                                      [1e-4, 1e-4, 1e-3, 1e-6, 1e-6])

    def test_the_other_steps_are_fractions_of_the_values(self):
        optimizer = self.optimizer(LorenzMie)
        optimizer.variables = ['z_p', 'a_p', 'k_p']
        values = self.evaluated(optimizer)
        np.testing.assert_allclose(optimizer._steps(values),
                                   [153e-5, 0.78e-5, np.sqrt(2.**-52)])

    def test_a_particle_of_several_takes_the_step_without_the_number(self):
        optimizer = self.optimizer()
        optimizer._variables = ['z_p1', 'a_p2']
        np.testing.assert_array_equal(
            optimizer._steps(np.array([100., 1.])), [1e-3, 1e-6])

    def test_the_jacobian_is_the_derivative(self):
        optimizer = self.optimizer()
        values = self.evaluated(optimizer)
        optimizer.model.coordinates = self.hologram.flat_coordinates
        jacobian = optimizer._difference_jac(values)
        for index, name in enumerate(optimizer.variables):
            with self.subTest(name=name):
                step = np.zeros(values.size)
                step[index] = optimizer._steps(values)[index]
                central = (optimizer._residuals(values + step)
                           - optimizer._residuals(values - step))
                central /= 2. * step[index]
                np.testing.assert_allclose(jacobian[:, index],
                                           central,
                                           atol=2e-2 * np.abs(central).max())

    def test_the_residuals_are_not_computed_twice(self):
        optimizer = self.optimizer()
        values = self.evaluated(optimizer)
        optimizer.model.coordinates = self.hologram.flat_coordinates
        optimizer._residuals(values)
        with patch.object(optimizer.model,
                          'hologram',
                          wraps=optimizer.model.hologram) as hologram:
            optimizer._difference_jac(values)
        self.assertEqual(hologram.call_count, values.size)

    def test_a_step_does_not_leave_a_bound(self):
        optimizer = self.optimizer()
        optimizer.bounds = {'a_p': (0.5, 0.78)}     # the start is on it
        values = self.evaluated(optimizer)
        optimizer.model.coordinates = self.hologram.flat_coordinates
        seen = []
        residuals = optimizer._residuals

        def record(moved):
            seen.append(moved[3])
            return residuals(moved)

        with patch.object(optimizer, '_residuals', record):
            optimizer._difference_jac(values)
        self.assertLessEqual(max(seen), 0.78)
        self.assertLess(min(seen), 0.78)

    def test_the_fit_is_that_of_the_default_differences(self):
        stepped = self.optimizer().optimize(self.hologram)
        default = self.optimizer(LorenzMie).optimize(self.hologram)
        for name in ('x_p', 'y_p', 'z_p', 'a_p', 'n_p'):
            with self.subTest(name=name):
                self.assertAlmostEqual(stepped[name],
                                       default[name],
                                       delta=0.1 * default['d' + name])
                self.assertAlmostEqual(stepped['d' + name],
                                       default['d' + name],
                                       delta=0.02 * default['d' + name])

    def test_the_model_is_left_at_the_fit(self):
        for cls in (LorenzMie, _Stepped):
            for method in ('lm', 'trf'):
                with self.subTest(cls=cls.__name__, method=method):
                    optimizer = self.optimizer(cls)
                    optimizer.settings['method'] = method
                    result = optimizer.optimize(self.hologram)
                    for name in optimizer.variables:
                        self.assertEqual(optimizer.model.properties[name],
                                         result[name])

    def test_a_jacobian_that_is_asked_for_is_kept(self):
        optimizer = self.optimizer()
        optimizer.settings['jac'] = '3-point'
        with patch.object(optimizer, '_difference_jac') as ours:
            optimizer.optimize(self.hologram)
        ours.assert_not_called()


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
