'''Tests of the lmtool window, without a display.

They need a Qt binding (``pip install pylorenzmie[gui]``) and are skipped
without one.  :class:`LMTool` is created with a file name: without one it
opens a modal file dialog and waits for a person.
'''
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

try:
    from pyqtgraph import mkQApp
    from pylorenzmie.lmtool.ALMWidget import ALMWidget
    from pylorenzmie.lmtool.LMTool import LMTool
    from pylorenzmie.lmtool.LMWidget import LMWidget
    from pylorenzmie.lmtool.tasks import Cancelled
    _qt = True
except Exception:       # no Qt binding, or no usable platform plugin
    _qt = False

IMAGE = Path(__file__).parents[1] / 'docs' / 'tutorials' / 'crop.png'


def setUpModule():
    if _qt:
        mkQApp()


class ToolTestCase(unittest.TestCase):
    '''A window on the tutorial hologram, the search for particles done.'''

    @staticmethod
    def wait(tool, seconds=60):
        '''Run the event loop until the tasks of the tool are done.'''
        end = time.time() + seconds
        while tool.tasks.busy and time.time() < end:
            mkQApp().processEvents()
            time.sleep(0.005)
        mkQApp().processEvents()

    def tool(self, panel=None):
        tool = LMTool(panel or ALMWidget, str(IMAGE))
        self.addCleanup(tool.close)
        self.wait(tool)
        return tool

    def slowly(self, name):
        '''Make the task ``name`` wait until it is cancelled.'''
        started = threading.Event()

        def slow(snapshot, cancel):
            started.set()
            cancel.wait(30)
            raise Cancelled

        patcher = mock.patch(f'pylorenzmie.lmtool.LMTool.run_{name}', slow)
        patcher.start()
        self.addCleanup(patcher.stop)
        return started


@unittest.skipUnless(_qt, 'a Qt binding is not available')
class TestSession(ToolTestCase):

    def test_a_session(self):
        tool = self.tool()
        self.assertEqual(tool._data.shape, (390, 390))
        # the ROI controls the position of the model
        tool.imageWidget.x_p, tool.imageWidget.y_p = 195.5, 193.2
        properties = tool.controls.properties
        self.assertAlmostEqual(properties['x_p'], 195.5, delta=0.1)
        self.assertAlmostEqual(properties['y_p'], 193.2, delta=0.1)
        # a fit changes the shown parameters and not the hidden ones
        before = tool.session.properties
        tool.optimize()
        self.wait(tool)
        for name in ('defocus', 'coma', 'astigmatism', 'distortion',
                     'x_c', 'y_c', 'spherical', 'piston'):
            self.assertEqual(tool.session.get(name), 0.)
        self.assertNotEqual(tool.session.get('z_p'), before['z_p'])
        self.assertEqual(tool.particles.results().iloc[0]['success'], True)

    def test_the_noise_is_estimated_when_an_image_is_loaded(self):
        tool = self.tool()
        noise = tool.controls.properties['noise']    # not the default 0.05
        self.assertAlmostEqual(noise, 0.035, delta=0.005)
        self.assertAlmostEqual(tool.session.model.instrument.noise, noise,
                               delta=1e-3)

    def test_a_panel_without_a_noise_control_still_gets_the_estimate(self):
        tool = self.tool(LMWidget)
        self.assertAlmostEqual(tool.session.model.instrument.noise, 0.035,
                               delta=0.005)

    def test_estimate_noise_uses_the_region_of_interest(self):
        tool = self.tool()
        image = 1. + np.random.default_rng(0).normal(0., 0.1, (390, 390))
        image[:195] = 1. + np.random.default_rng(1).normal(0., 0.02,
                                                           (195, 390))
        tool.data = image
        tool.imageWidget.setRegion(200., 300., 60)   # the noisy half
        tool.estimateNoise()
        self.assertAlmostEqual(tool.controls.properties['noise'], 0.1,
                               delta=0.01)
        self.assertIn('Noise: 0.1', tool.statusBar().currentMessage())


@unittest.skipUnless(_qt, 'a Qt binding is not available')
class TestTasks(ToolTestCase):

    def test_cancel_stops_a_fit_and_changes_nothing(self):
        tool = self.tool()
        started = self.slowly('fit')
        before = tool.session.properties
        self.assertFalse(tool.actionCancel.isEnabled())
        tool.optimize()
        self.assertTrue(tool.tasks.busy)
        self.assertTrue(tool.actionCancel.isEnabled())
        self.assertFalse(tool.actionOptimize.isEnabled())
        self.assertTrue(started.wait(10))
        tool.actionCancel.trigger()
        self.wait(tool)
        self.assertFalse(tool.tasks.busy)
        self.assertEqual(tool.statusBar().currentMessage(), 'Cancelled')
        self.assertEqual(tool.session.properties, before)
        self.assertTrue(tool.actionOptimize.isEnabled())
        self.assertFalse(tool.actionCancel.isEnabled())
        self.assertEqual(len(tool.particles.results()), 0)

    def test_cancel_stops_an_estimate_and_leaves_nothing_to_undo(self):
        tool = self.tool()
        started = self.slowly('estimate')
        tool.estimate()
        self.assertTrue(started.wait(10))
        tool.tasks.cancel()
        self.wait(tool)
        self.assertFalse(tool.actionUndoEstimate.isEnabled())
        self.assertTrue(tool.actionEstimate.isEnabled())

    def test_a_failed_fit_is_reported(self):
        tool = self.tool()

        def broken(snapshot, cancel):
            raise ValueError('no good')

        with mock.patch('pylorenzmie.lmtool.LMTool.run_fit', broken):
            with self.assertLogs(level='ERROR'):
                tool.optimize()
                self.wait(tool)
        self.assertIn('Optimization failed: ValueError: no good',
                      tool.statusBar().currentMessage())
        self.assertTrue(tool.actionOptimize.isEnabled())

    def test_a_task_is_not_started_while_another_runs(self):
        tool = self.tool()
        started = self.slowly('fit')
        tool.optimize()
        self.assertTrue(started.wait(10))
        tool.optimize()                       # ignored, and says why
        self.assertIn('Busy', tool.statusBar().currentMessage())
        tool.estimate()                       # ignored
        tool.tasks.cancel()
        self.wait(tool)
        self.assertFalse(tool.actionUndoEstimate.isEnabled())

    def test_escape_cancels(self):
        tool = self.tool()
        self.assertEqual(tool.actionCancel.shortcut().toString(), 'Esc')


@unittest.skipUnless(_qt, 'a Qt binding is not available')
class TestParticles(ToolTestCase):

    def test_the_roi_starts_on_the_particle(self):
        tool = self.tool()
        properties = tool.controls.properties
        self.assertAlmostEqual(properties['x_p'], 195.5, delta=2.)
        self.assertAlmostEqual(properties['y_p'], 193.2, delta=2.)
        self.assertGreaterEqual(tool.imageWidget.radius, 30)
        self.assertEqual(tool.particles.count, 1)
        self.assertIn('Only one particle', tool.statusBar().currentMessage())

    def test_the_window_stays_usable_while_it_looks(self):
        tool = LMTool(ALMWidget, str(IMAGE))
        self.addCleanup(tool.close)
        if tool.tasks.busy:                    # the search is running
            self.assertEqual(tool.tasks.name, 'localize')
            self.assertTrue(tool.actionOptimize.isEnabled())
            self.assertTrue(tool.actionCancel.isEnabled())
        self.wait(tool)

    def test_next_particle_cycles_through_the_checked_particles(self):
        tool = self.tool()
        tool.particles.setFound(pd.DataFrame(
            [dict(x_p=100., y_p=120., bbox=((50, 70), 100, 100)),
             dict(x_p=250., y_p=200., bbox=((190, 140), 120, 120)),
             dict(x_p=300., y_p=300., bbox=((250, 250), 100, 100))]))
        tool.nextParticle()
        self.assertAlmostEqual(tool.controls.properties['x_p'], 100., delta=1.)
        tool.nextParticle()
        self.assertAlmostEqual(tool.controls.properties['x_p'], 250., delta=1.)
        self.assertEqual(tool.imageWidget.radius, 60)
        tool.nextParticle()
        tool.nextParticle()                     # and around again
        self.assertAlmostEqual(tool.controls.properties['y_p'], 120., delta=1.)

    def test_a_particle_that_is_not_checked_is_skipped(self):
        from pyqtgraph.Qt.QtCore import Qt
        tool = self.tool()
        tool.particles.setFound(pd.DataFrame(
            [dict(x_p=100., y_p=120., bbox=((50, 70), 100, 100)),
             dict(x_p=250., y_p=200., bbox=((190, 140), 120, 120)),
             dict(x_p=300., y_p=300., bbox=((250, 250), 100, 100))]))
        tool.particles.setData(tool.particles.index(1, 0),
                               Qt.CheckState.Unchecked,
                               Qt.ItemDataRole.CheckStateRole)
        tool.nextParticle()
        tool.nextParticle()
        self.assertAlmostEqual(tool.controls.properties['x_p'], 300., delta=1.)

    def test_no_particle_checked(self):
        from pyqtgraph.Qt.QtCore import Qt
        tool = self.tool()
        tool.particles.setData(tool.particles.index(0, 0),
                               Qt.CheckState.Unchecked,
                               Qt.ItemDataRole.CheckStateRole)
        tool.nextParticle()
        self.assertIn('No particle is checked',
                      tool.statusBar().currentMessage())

    def test_many_candidates_are_limited_and_ranked(self):
        tool = self.tool()
        noise = np.random.default_rng(1).normal(1., 0.03, (400, 400))
        tool.data = noise
        tool.particles.clear()
        tool.findParticles()
        self.wait(tool)
        self.assertLessEqual(tool.particles.count, tool.MAX_CANDIDATES)
        self.assertGreater(tool._nfound, tool.particles.count)
        self.assertIn('noisy', tool.statusBar().currentMessage())

    def test_no_particles_is_reported(self):
        tool = self.tool()
        tool.particles.clear()
        tool.data = np.ones((200, 200))
        tool.findParticles()
        self.wait(tool)
        self.assertIn('No particles', tool.statusBar().currentMessage())

    def test_a_new_image_replaces_a_search_that_is_running(self):
        tool = LMTool(ALMWidget, str(IMAGE))
        self.addCleanup(tool.close)
        started = threading.Event()
        calls = []

        def slow(request, cancel):
            calls.append(request)
            started.set()
            if len(calls) == 1:
                cancel.wait(30)
                raise Cancelled
            return pd.DataFrame(columns=['x_p', 'y_p', 'bbox'])

        self.wait(tool)
        with mock.patch('pylorenzmie.lmtool.LMTool.run_localize', slow):
            tool.findParticles()
            self.assertTrue(started.wait(10))
            tool.findParticles()               # another image arrives
            self.wait(tool)
        self.assertEqual(len(calls), 2)

    def test_a_particle_can_be_added_from_the_roi_and_removed(self):
        tool = self.tool()
        tool.imageWidget.setRegion(300., 100., 50)
        tool.addParticle()
        self.assertEqual(tool.particles.count, 2)
        entry = tool.particles.entry(tool.particles.current)
        self.assertEqual(entry['source'], 'added')
        self.assertAlmostEqual(entry['x_p'], 300., delta=1.)
        self.assertEqual(entry['radius'], 50)
        tool.removeParticle()
        self.assertEqual(tool.particles.count, 1)

    def test_added_particles_survive_a_new_search(self):
        tool = self.tool()
        tool.imageWidget.setRegion(300., 100., 50)
        tool.addParticle()
        tool.findParticles()
        self.wait(tool)
        self.assertEqual(sorted(e['source'] for e in
                                (tool.particles.entry(r)
                                 for r in range(tool.particles.count))),
                         ['added', 'found'])

    def test_a_new_image_forgets_the_particles(self):
        tool = self.tool()
        tool.imageWidget.setRegion(300., 100., 50)
        tool.addParticle()
        tool.readHologram(str(IMAGE))
        self.wait(tool)
        self.assertEqual(tool.particles.count, 1)    # only what was found

    def test_selecting_a_row_moves_the_roi(self):
        tool = self.tool()
        tool.imageWidget.setRegion(300., 100., 50)
        tool.addParticle()
        tool.particleView.selectRow(0)
        self.assertAlmostEqual(tool.controls.properties['x_p'], 195.5,
                               delta=2.)
        tool.particleView.selectRow(1)
        self.assertAlmostEqual(tool.controls.properties['x_p'], 300.,
                               delta=1.)
        self.assertEqual(tool.imageWidget.radius, 50)


@unittest.skipUnless(_qt, 'a Qt binding is not available')
class TestResults(ToolTestCase):

    def test_a_fit_is_recorded_for_the_particle_it_was_made_on(self):
        tool = self.tool()
        tool.estimate()
        self.wait(tool)
        entry = tool.particles.entry(0)
        self.assertEqual(entry['status'], 'estimated')
        self.assertAlmostEqual(entry['properties']['z_p'],
                               tool.session.get('z_p'))
        tool.optimize()
        self.wait(tool)
        entry = tool.particles.entry(0)
        self.assertEqual(entry['status'], 'fitted')
        self.assertTrue(entry['result']['success'])
        self.assertEqual(tool.particles.count, 1)

    def test_a_fit_away_from_every_particle_adds_one(self):
        tool = self.tool()
        tool.imageWidget.setRegion(120., 280., 60)
        tool.optimize()
        self.wait(tool)
        self.assertEqual(tool.particles.count, 2)
        self.assertEqual(tool.particles.entry(1)['source'], 'added')
        self.assertEqual(tool.particles.entry(1)['status'], 'fitted')

    def test_the_results_have_a_row_for_each_fitted_particle(self):
        tool = self.tool()
        tool.optimize()
        self.wait(tool)
        results = tool.particles.results()
        self.assertEqual(list(results.index), [0])
        for name in ('x_p', 'z_p', 'a_p', 'n_p', 'dz_p', 'redchi'):
            self.assertIn(name, results.columns)
        self.assertEqual(results.index.name, 'particle')

    def test_a_particle_that_is_not_checked_is_not_in_the_results(self):
        from pyqtgraph.Qt.QtCore import Qt
        tool = self.tool()
        tool.optimize()
        self.wait(tool)
        tool.particles.setData(tool.particles.index(0, 0),
                               Qt.CheckState.Unchecked,
                               Qt.ItemDataRole.CheckStateRole)
        self.assertEqual(len(tool.particles.results()), 0)
        self.assertEqual(len(tool.particles.results(included=False)), 1)

    def test_selecting_a_fitted_particle_restores_its_fit(self):
        tool = self.tool()
        tool.optimize()
        self.wait(tool)
        fitted = tool.session.get('z_p')
        tool.imageWidget.setRegion(300., 100., 50)
        tool.addParticle()
        tool.session.update(dict(z_p=50.))
        tool.particleView.selectRow(0)
        self.assertAlmostEqual(tool.session.get('z_p'), fitted)

    def test_saving_writes_one_row_per_fitted_particle(self):
        tool = self.tool()
        tool.optimize()
        self.wait(tool)
        with tempfile.TemporaryDirectory() as directory:
            filename = str(Path(directory) / 'results.json')
            tool.saveResults(filename)
            saved = json.loads(Path(filename).read_text())
        self.assertIn('Saved 1 particle', tool.statusBar().currentMessage())
        self.assertEqual(len(saved['results']), 1)
        self.assertIn('z_p', saved['results'][0])
        self.assertIn('wavelength', saved['metadata'])
        self.assertEqual(saved['metadata']['datafile'], str(IMAGE))

    def test_saving_with_nothing_fitted_says_so(self):
        tool = self.tool()
        tool.saveResults('/nonexistent/never.json')
        self.assertIn('nothing to save', tool.statusBar().currentMessage())

    def test_a_failure_to_save_is_reported(self):
        tool = self.tool()
        tool.optimize()
        self.wait(tool)
        with mock.patch('pylorenzmie.lmtool.LMTool.save_results',
                        side_effect=OSError('disk full')):
            with self.assertLogs(level='ERROR'):
                tool.saveResults('x.json')
        self.assertIn('disk full', tool.statusBar().currentMessage())


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
