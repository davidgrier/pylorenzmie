import importlib
import unittest
from unittest.mock import patch

import numpy as np

from pylorenzmie.analysis import Mask, Optimizer
from pylorenzmie.analysis.broyden import broyden_least_squares
from pylorenzmie.analysis.Hologram import Hologram
from pylorenzmie.theory import LorenzMie

# the package namespace holds the class of the same name
sphere = importlib.import_module('pylorenzmie.theory.Sphere')

NAMES = ('x_p', 'y_p', 'z_p', 'a_p', 'n_p')


class Decay:
    '''Residuals of a decaying exponential on an offset, with noise.'''

    truth = np.array([2.0, 1.3, 0.4])

    def __init__(self, noise=0.01):
        self.t = np.linspace(0., 4., 200)
        self.noise = noise
        error = np.random.default_rng(0).normal(0., noise, self.t.size)
        self.data = self.model(self.truth) + error
        self.calls = 0
        self.jacobians = 0

    def model(self, x):
        return x[0] * np.exp(-x[1] * self.t) + x[2]

    def fun(self, x):
        self.calls += 1
        return (self.model(x) - self.data) / (self.noise or 1.)

    def jac(self, x, residuals):
        self.jacobians += 1
        decay = np.exp(-x[1] * self.t)
        columns = [decay, -x[0] * self.t * decay, np.ones_like(self.t)]
        return np.column_stack(columns) / (self.noise or 1.)


class TestBroydenLeastSquares(unittest.TestCase):

    start = np.array([1.5, 0.8, 0.2])

    def solve(self, problem, **kwargs):
        return broyden_least_squares(problem.fun,
                                     self.start,
                                     problem.jac,
                                     **kwargs)

    def test_the_minimum_is_found(self):
        problem = Decay()
        fit = self.solve(problem, tol=1e-6)
        self.assertTrue(fit.success)
        self.assertEqual(fit.status, 1)
        # the gradient vanishes at the minimum, in units of the noise
        J = problem.jac(fit.x, fit.fun)
        newton = np.linalg.lstsq(J, fit.fun, rcond=None)[0]
        sigma = np.sqrt(np.diag(np.linalg.inv(J.T @ J)))
        np.testing.assert_array_less(np.abs(newton), 0.01 * sigma)

    def test_the_result_is_that_of_least_squares(self):
        problem = Decay()
        fit = self.solve(problem)
        np.testing.assert_array_equal(fit.fun, problem.fun(fit.x))
        np.testing.assert_array_equal(fit.jac, problem.jac(fit.x, fit.fun))
        self.assertAlmostEqual(fit.cost, 0.5 * fit.fun @ fit.fun)

    def test_the_calls_are_counted(self):
        problem = Decay()
        fit = self.solve(problem)
        self.assertEqual(fit.nfev, problem.calls)
        self.assertEqual(fit.njev, problem.jacobians)

    def test_the_jacobian_is_not_computed_at_every_step(self):
        problem = Decay()
        fit = self.solve(problem, tol=1e-6)
        self.assertLess(fit.njev, fit.nfev)

    def test_a_looser_tolerance_stops_sooner(self):
        loose = self.solve(Decay(), tol=10.)
        tight = self.solve(Decay(), tol=1e-8)
        self.assertLess(loose.nfev + loose.njev, tight.nfev + tight.njev)
        self.assertLessEqual(tight.cost, loose.cost)

    def test_the_budget_is_kept(self):
        problem = Decay()
        fit = self.solve(problem, max_nfev=3)
        self.assertEqual(fit.nfev, 3)
        self.assertEqual(fit.status, 0)
        self.assertFalse(fit.success)
        np.testing.assert_array_equal(fit.jac, problem.jac(fit.x, fit.fun))

    def test_data_without_noise(self):
        problem = Decay(noise=0.)
        fit = self.solve(problem)
        self.assertTrue(fit.success)
        np.testing.assert_allclose(fit.x, problem.truth, rtol=1e-6)

    def test_points_that_must_not_be_accepted(self):
        for rejected in (np.inf, np.nan, 1e6):
            with self.subTest(rejected=rejected):
                problem = Decay()
                fun = problem.fun

                def limited(x, fun=fun, rejected=rejected):
                    residuals = fun(x)
                    if x[1] > 1.35:         # beyond the minimum
                        return np.full_like(residuals, rejected)
                    return residuals

                fit = broyden_least_squares(limited,
                                            self.start,
                                            problem.jac)
                self.assertTrue(np.isfinite(fit.x).all())
                self.assertLessEqual(fit.x[1], 1.35)
                self.assertTrue(fit.success)

    def test_no_step_that_helps(self):
        start = self.start

        def only_here(x):
            return np.ones(5) if np.array_equal(x, start) else np.full(5, 2.)

        fit = broyden_least_squares(only_here,
                                    start,
                                    lambda x, r: np.eye(5, 3))
        self.assertEqual(fit.status, 3)
        self.assertFalse(fit.success)
        np.testing.assert_array_equal(fit.x, start)

    def test_noise_that_the_jacobian_does_not_see(self):
        # a model that is only known to a few digits, as in single
        # precision, and a Jacobian that is wrong by as much
        problem = Decay()
        rng = np.random.default_rng(1)

        def coarse(x):
            residuals = problem.fun(x)
            return residuals + rng.normal(0., 0.002, residuals.size)

        def rough(x, residuals):
            J = problem.jac(x, residuals)
            return J * (1. + rng.normal(0., 0.1, J.shape))

        fit = broyden_least_squares(coarse, self.start, rough, tol=1e-4)
        self.assertEqual(fit.status, 4)
        self.assertTrue(fit.success)
        self.assertLess(fit.nfev, 40)
        best = self.solve(Decay(), tol=1e-8)
        sigma = np.sqrt(np.diag(np.linalg.inv(best.jac.T @ best.jac)))
        np.testing.assert_array_less(np.abs(fit.x - best.x), sigma)

    def test_a_parameter_without_effect(self):
        problem = Decay()

        def jac(x, residuals):
            J = problem.jac(x, residuals)
            return np.column_stack([J, np.zeros(J.shape[0])])

        start = np.append(self.start, 7.)
        fit = broyden_least_squares(lambda x: problem.fun(x[:3]), start, jac)
        self.assertTrue(fit.success)
        self.assertEqual(fit.x[3], 7.)

    def test_the_tolerance_is_positive(self):
        with self.assertRaises(ValueError):
            self.solve(Decay(), tol=0.)


class TestBroydenOptimizer(unittest.TestCase):

    shape = (120, 120)

    def setUp(self):
        self.truth = {'x_p': 60.3, 'y_p': 58.6, 'z_p': 200.,
                      'a_p': 0.8, 'n_p': 1.45}
        data = self.model(self.truth).hologram().reshape(self.shape)
        data += np.random.default_rng(0).normal(0., 0.03, self.shape)
        self.hologram = Hologram(data)

    def model(self, properties):
        model = LorenzMie(coordinates=LorenzMie.meshgrid(self.shape))
        model.instrument.noise = 0.03
        model.properties = properties
        return model

    def optimizer(self, method='broyden', **kwargs):
        start = {'x_p': 60., 'y_p': 59., 'z_p': 205.,
                 'a_p': 0.77, 'n_p': 1.465}
        optimizer = Optimizer(model=self.model(start),
                              mask=Mask(fraction=0.25, seed=0),
                              **kwargs)
        optimizer.settings['method'] = method
        return optimizer

    def test_the_fit_agrees_with_levenberg_marquardt(self):
        ours = self.optimizer().optimize(self.hologram)
        theirs = self.optimizer('lm').optimize(self.hologram)
        self.assertTrue(ours.success)
        for name in NAMES:
            with self.subTest(name=name):
                error = theirs['d' + name]
                self.assertAlmostEqual(ours[name],
                                       theirs[name],
                                       delta=0.1 * error)
                self.assertAlmostEqual(ours['d' + name],
                                       error,
                                       delta=0.02 * error)
        self.assertAlmostEqual(ours.redchi, theirs.redchi, delta=1e-3)
        self.assertEqual(ours.npix, theirs.npix)

    def test_it_evaluates_the_model_less_often(self):
        counts = []
        for method in ('broyden', 'lm'):
            optimizer = self.optimizer(method)
            with patch.object(optimizer.model,
                              'hologram',
                              wraps=optimizer.model.hologram) as hologram:
                optimizer.optimize(self.hologram)
            counts.append(hologram.call_count)
        self.assertLess(counts[0], counts[1])

    def test_the_model_is_left_at_the_fit(self):
        optimizer = self.optimizer()
        result = optimizer.optimize(self.hologram)
        for name in NAMES:
            self.assertEqual(optimizer.model.properties[name], result[name])

    def test_the_tolerance_is_a_setting(self):
        optimizer = self.optimizer()
        optimizer.settings['chi2tol'] = 1e-4
        with patch('pylorenzmie.analysis.Optimizer.broyden_least_squares',
                   wraps=broyden_least_squares) as solver:
            optimizer.optimize(self.hologram)
        self.assertEqual(solver.call_args.kwargs['tol'], 1e-4)
        self.assertEqual(solver.call_args.kwargs['max_nfev'], 2000)
        self.assertEqual(optimizer.settings['chi2tol'], 1e-4)
        self.assertEqual(optimizer.metadata['chi2tol'], 1e-4)

    def test_the_tolerance_does_not_disturb_the_other_methods(self):
        optimizer = self.optimizer('lm')
        optimizer.settings['chi2tol'] = 1e-4
        self.assertTrue(optimizer.optimize(self.hologram).success)

    def test_the_budget_is_that_of_the_settings(self):
        optimizer = self.optimizer()
        optimizer.settings['max_nfev'] = 2
        result = optimizer.optimize(self.hologram)
        self.assertFalse(result.success)

    def test_what_it_does_not_do_is_refused(self):
        cases = ({'loss': 'cauchy'}, {'x_scale': 1.}, {'jac': '3-point'})
        for settings in cases:
            with self.subTest(settings=settings):
                optimizer = self.optimizer()
                optimizer.settings.update(settings)
                with self.assertRaisesRegex(ValueError, 'broyden'):
                    optimizer.optimize(self.hologram)
        optimizer = self.optimizer(bounds={'a_p': (0.5, 2.)})
        with self.assertRaisesRegex(ValueError, 'bounds'):
            optimizer.optimize(self.hologram)

    def test_a_robust_fit_changes_the_method(self):
        optimizer = self.optimizer()
        optimizer.robust = True
        self.assertEqual(optimizer.settings['method'], 'trf')
        self.assertTrue(optimizer.optimize(self.hologram).success)

    def test_a_step_out_of_the_domain_of_the_model(self):
        # the truth, 0.8 um, is beyond the largest radius that is computed
        optimizer = self.optimizer()
        with patch.object(sphere, 'MAX_RADIUS', 0.785):
            result = optimizer.optimize(self.hologram)
        self.assertLessEqual(result['a_p'], 0.785)
        self.assertTrue(np.isfinite(result.redchi))

    def test_an_analytical_jacobian_is_used(self):
        class Analytic(LorenzMie):
            jac_params = frozenset(NAMES)
            asked = 0

            def jac(self):
                Analytic.asked += 1
                columns = {}
                base = self.hologram()
                for name in NAMES:
                    value = self.properties[name]
                    self.properties = {name: value * (1. + 1e-7)}
                    columns[name] = ((self.hologram() - base)
                                     / (value * 1e-7))
                    self.properties = {name: value}
                return columns

        optimizer = self.optimizer()
        model = Analytic(coordinates=optimizer.model.coordinates)
        model.properties = optimizer.model.properties
        optimizer.model = model
        with patch.object(optimizer, '_difference_jac') as differences:
            result = optimizer.optimize(self.hologram)
        differences.assert_not_called()
        self.assertGreater(Analytic.asked, 0)
        self.assertTrue(result.success)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
