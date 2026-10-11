import unittest

from pylorenzmie.analysis import Optimizer
from pylorenzmie.analysis.Hologram import Hologram
from pylorenzmie.analysis.Mask import Mask
from pylorenzmie.theory import LorenzMie
try:
    from pylorenzmie.theory.jaxLorenzMie import jaxLorenzMie
    _jax_available = True
except Exception:
    _jax_available = False
from pylorenzmie.lib import LMObject
from pathlib import Path
import cv2
import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).parent.resolve()
TEST_IMAGE = str(THIS_DIR / 'data' / 'crop.png')


class TestOptimizer(unittest.TestCase):

    def setUp(self):
        img = cv2.imread(TEST_IMAGE, cv2.IMREAD_GRAYSCALE).astype(float)
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
        img = cv2.imread(TEST_IMAGE, cv2.IMREAD_GRAYSCALE).astype(float)
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
        self.assertNotIn('jac', kwargs,
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


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
