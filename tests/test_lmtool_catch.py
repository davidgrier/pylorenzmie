'''Tests of the optional use of CATCH in lmtool.

The tests of the window need a Qt binding and run without a display; those
of the networks need CATCH, and are skipped without it.
'''
import importlib.machinery
import importlib.util
import os
import threading
import unittest
from unittest import mock

import numpy as np
import pandas as pd

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

try:
    import qtpy     # chooses the Qt binding (QT_API) before pyqtgraph can
    from pyqtgraph import mkQApp
    from pylorenzmie.analysis import catch_available
    from pylorenzmie.lmtool.tasks import (LocalizeRequest, Snapshot,
                                          run_estimate, run_localize)
    from pylorenzmie.theory import LorenzMie
    from tests.test_catchanalyzer import StandIn, analyzer, detections
    from tests.test_lmtool import ToolTestCase
    _qt = True
except Exception:       # no Qt binding, or no usable platform plugin
    _qt = False
    ToolTestCase = unittest.TestCase

_spec = importlib.util.find_spec('CATCH')
_catch = _spec is not None and _spec.origin is not None
INSTRUMENT = dict(wavelength=0.447, magnification=0.048, n_m=1.34)


def setUpModule():
    if _qt:
        mkQApp()


@unittest.skipUnless(_qt, 'a Qt binding is not available')
class TestCatchAction(ToolTestCase):

    def test_a_directory_named_catch_is_not_catch(self):
        spec = importlib.machinery.ModuleSpec
        namespace = spec('CATCH', None, is_package=True)
        installed = spec('CATCH', None, origin='__init__.py')
        for found, available in ((None, False),
                                 (namespace, False),
                                 (installed, True)):
            with mock.patch('importlib.util.find_spec', return_value=found):
                self.assertEqual(catch_available(), available)

    def installed(self):
        '''A window that finds CATCH, which finds nothing.'''
        nothing = pd.DataFrame(columns=['x_p', 'y_p', 'bbox'])
        with mock.patch('pylorenzmie.lmtool.LMTool.catch_available',
                        return_value=True):
            with mock.patch('pylorenzmie.lmtool.LMTool.run_localize',
                            return_value=nothing):
                return self.tool()

    def test_it_is_used_when_catch_is_installed(self):
        self.assertEqual(catch_available(), _catch)
        tool = self.installed()
        self.assertTrue(tool.actionCATCH.isEnabled())
        self.assertTrue(tool.useCATCH)
        tool.actionCATCH.setChecked(False)     # the user may do without
        self.assertFalse(tool.useCATCH)

    def test_it_is_disabled_when_catch_is_not_installed(self):
        with mock.patch('pylorenzmie.lmtool.LMTool.catch_available',
                        return_value=False):
            tool = self.tool()
        self.assertFalse(tool.actionCATCH.isEnabled())
        self.assertIn('not installed', tool.actionCATCH.toolTip())
        tool.actionCATCH.setChecked(True)
        self.assertFalse(tool.useCATCH)

    def test_find_particles_tells_the_task_to_use_catch(self):
        tool = self.installed()
        requests = []

        def localize(request, cancel):
            requests.append(request)
            return pd.DataFrame(columns=['x_p', 'y_p', 'bbox'])

        with mock.patch('pylorenzmie.lmtool.LMTool.run_localize', localize):
            tool.findParticles()
            self.wait(tool)
            tool.actionCATCH.setChecked(False)
            tool.findParticles()
            self.wait(tool)
        self.assertEqual([r.catch for r in requests], [True, False])
        self.assertEqual(requests[0].instrument, INSTRUMENT)

    def test_estimate_tells_the_task_to_use_catch(self):
        tool = self.installed()
        snapshots = []

        def estimate(snapshot, cancel):
            snapshots.append(snapshot)
            return pd.Series(dict(z_p=300., a_p=1., n_p=1.45))

        with mock.patch('pylorenzmie.lmtool.LMTool.run_estimate', estimate):
            tool.estimate()
            self.wait(tool)
            tool.actionCATCH.setChecked(False)
            tool.estimate()
            self.wait(tool)
        self.assertEqual([s.catch for s in snapshots], [True, False])

    def test_a_warning_of_catch_is_shown(self):
        tool = self.installed()
        found = pd.DataFrame(columns=['x_p', 'y_p', 'bbox'])
        found.attrs['warning'] = 'CATCH may not suit these pixels'
        with mock.patch('pylorenzmie.lmtool.LMTool.run_localize',
                        return_value=found):
            with self.assertLogs(level='WARNING'):
                tool.findParticles()
                self.wait(tool)
        self.assertEqual(tool.statusBar().currentMessage(),
                         found.attrs['warning'])


@unittest.skipUnless(_qt, 'a Qt binding is not available')
class TestCatchTasksStandIn(unittest.TestCase):

    def run_with(self, found, request):
        app = StandIn(found=found)
        with mock.patch.object(analyzer, '_app', return_value=app):
            return app, run_localize(request, threading.Event())

    def test_the_strongest_candidates_come_first(self):
        found = detections([0.3, 0.9, 0.6])
        saturated = np.zeros((64, 64), bool)
        image = np.ones((64, 64))
        request = LocalizeRequest(image, 2, True, INSTRUMENT, saturated)
        app, result = self.run_with(found, request)
        self.assertEqual(list(result.confidence), [0.9, 0.6])
        self.assertEqual(result.attrs['nfound'], 3)
        self.assertEqual(result.attrs['warning'], '')
        self.assertIs(app.saturation, saturated)
        self.assertEqual(app.instrument, INSTRUMENT)

    def test_finding_nothing_gives_an_empty_table(self):
        request = LocalizeRequest(np.ones((64, 64)), 10, True, INSTRUMENT)
        _, result = self.run_with(pd.DataFrame(), request)
        self.assertEqual(len(result), 0)
        self.assertIn('bbox', result.columns)


@unittest.skipUnless(_qt and _catch, 'CATCH is not installed')
class TestCatchTasks(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        model = LorenzMie(coordinates=LorenzMie.meshgrid((400, 400)))
        model.particle.r_p = [200., 200., 300.]
        model.particle.a_p, model.particle.n_p = 1.0, 1.45
        noise = np.random.default_rng(0).normal(0., 0.02, (400, 400))
        cls.image = model.hologram().reshape(400, 400) + noise

    def test_it_finds_the_particle(self):
        found = run_localize(LocalizeRequest(self.image, 10, True, INSTRUMENT),
                             threading.Event())
        self.assertEqual(len(found), 1)
        self.assertAlmostEqual(found.iloc[0]['x_p'], 200., delta=3.)
        self.assertAlmostEqual(found.iloc[0]['y_p'], 200., delta=3.)
        self.assertGreater(found.iloc[0]['bbox'][1], 0)
        self.assertEqual(found.attrs['nfound'], 1)

    def test_it_estimates_the_particle_in_a_region(self):
        properties = dict(LorenzMie().properties)
        properties.update(x_p=200., y_p=200.)
        y, x = np.mgrid[100:300, 100:300]
        snapshot = Snapshot.take(LorenzMie,
                                 properties,
                                 self.image[100:300, 100:300],
                                 np.array([x, y], float),
                                 catch=True)
        estimate = run_estimate(snapshot, threading.Event())
        self.assertAlmostEqual(estimate['a_p'], 1.0, delta=0.1)
        self.assertAlmostEqual(estimate['n_p'], 1.45, delta=0.05)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
