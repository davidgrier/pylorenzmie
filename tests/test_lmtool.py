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

import cv2
import numpy as np
import pandas as pd

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

try:
    from pyqtgraph import mkQApp
    from pylorenzmie.lmtool.ALMWidget import ALMWidget
    from pylorenzmie.lmtool.LMTool import LMTool
    from pylorenzmie.lmtool.LMWidget import LMWidget
    from pylorenzmie.lmtool.tasks import Cancelled
    from pylorenzmie.utilities import Normalizer
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

    def tool(self, panel=None, background=None):
        tool = LMTool(panel or ALMWidget, str(IMAGE), background)
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

    def test_a_region_past_the_edge_is_drawn_where_it_is(self):
        tool = self.tool()
        tool.imageWidget.setRegion(20., 195., 60)
        data, rect, coordinates = tool.crop()
        self.assertLess(tool.imageWidget.rect().x(), 0.)    # past the edge
        self.assertEqual(rect.x(), 0.)
        self.assertEqual((rect.width(), rect.height()), data.shape[::-1])
        self.assertEqual(coordinates.shape[1:], data.shape)


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

    def test_an_estimate_leaves_the_fixed_parameters_alone(self):
        tool = self.tool()
        tool.session.update(dict(a_p=1.234, n_p=1.456))
        tool.session.setFixed('n_p', True)
        tool.session.setFixed('a_p', False)
        tool.estimate()
        self.wait(tool)
        self.assertEqual(tool.session.get('n_p'), 1.456)
        self.assertNotEqual(tool.session.get('a_p'), 1.234)

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


@unittest.skipUnless(_qt, 'a Qt binding is not available')
class TestOpenDialogs(ToolTestCase):
    '''The dialogs for opening files start where the last choice was.'''

    def choose(self, tool, method, filename=''):
        '''Run ``method`` with a dialog that answers ``filename``; return
        the directory or file that the dialog was told to start at.'''
        with mock.patch('pylorenzmie.lmtool.LMTool.QFileDialog.'
                        'getOpenFileName',
                        return_value=(filename, '')) as dialog:
            method()
        return dialog.call_args.args[2]

    def test_a_hologram_starts_at_the_current_file(self):
        tool = self.tool()
        self.assertEqual(self.choose(tool, tool.readHologram), str(IMAGE))

    def test_a_hologram_starts_nowhere_without_a_file(self):
        tool = self.tool()
        tool._datafile = None
        self.assertEqual(self.choose(tool, tool.readHologram), '')

    def test_a_cancelled_dialog_keeps_the_current_file(self):
        tool = self.tool()
        self.choose(tool, tool.readHologram)
        self.assertEqual(tool._datafile, str(IMAGE))

    def test_a_background_starts_beside_the_hologram(self):
        tool = self.tool()
        self.assertEqual(self.choose(tool, tool.readBackground),
                         str(IMAGE.parent))

    def test_a_background_starts_at_the_last_background(self):
        tool = self.tool()
        self.choose(tool, tool.readBackground, str(IMAGE))
        self.assertEqual(tool.normalizer.source, str(IMAGE))
        self.assertEqual(self.choose(tool, tool.readBackground), str(IMAGE))

    def test_a_background_starts_nowhere_without_any_file(self):
        tool = self.tool()
        tool._datafile = None
        self.assertEqual(self.choose(tool, tool.readBackground), '')


@unittest.skipUnless(_qt, 'a Qt binding is not available')
class TestBackground(ToolTestCase):

    def shown(self, tool):
        '''The background that the window says it uses.'''
        return tool._background.text()

    def test_the_median_is_the_default(self):
        tool = self.tool()
        self.assertEqual(tool.normalizer.method, 'median')
        self.assertEqual(self.shown(tool), 'background: median')

    def test_a_number(self):
        tool = self.tool(background=100)
        self.assertEqual(tool.normalizer.method, 'reference')
        self.assertEqual(tool.normalizer.reference, 100.)
        self.assertIsNone(tool.normalizer.source)
        np.testing.assert_allclose(tool.data, tool._raw / 100.)
        self.assertEqual(self.shown(tool), 'background: 100')

    def test_a_number_given_as_text(self):
        self.assertEqual(self.tool(background='100').normalizer.reference,
                         100.)

    def test_a_file_at_startup_is_remembered_for_the_dialog(self):
        tool = self.tool(background=str(IMAGE))
        self.assertEqual(tool.normalizer.source, str(IMAGE))
        self.assertEqual(self.shown(tool), f'background: {IMAGE.name}')
        with mock.patch('pylorenzmie.lmtool.LMTool.QFileDialog.'
                        'getOpenFileName', return_value=('', '')) as dialog:
            tool.readBackground()
        self.assertEqual(dialog.call_args.args[2], str(IMAGE))

    def test_a_background_that_is_not_there_is_reported(self):
        with self.assertLogs(level='ERROR'):
            tool = self.tool(background='no-such-file.png')
        self.assertEqual(tool.normalizer.method, 'median')
        self.assertIn('not found', self.shown(tool))
        self.assertTrue(self.shown(tool).startswith('background: median'))

    def test_a_file_that_is_not_an_image_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'background.txt'
            path.write_text('not an image')
            with self.assertLogs(level='ERROR'):
                tool = self.tool(background=str(path))
        self.assertEqual(tool.normalizer.method, 'median')
        self.assertIn('Could not read', self.shown(tool))

    def test_a_failed_background_keeps_the_previous_one(self):
        tool = self.tool(background=100)
        with self.assertLogs(level='ERROR'):
            self.assertFalse(tool.setBackground('no-such-file.png'))
        self.assertEqual(tool.normalizer.reference, 100.)
        self.assertIn('background: 100', self.shown(tool))
        self.assertIn('not found', tool.statusBar().currentMessage())

    def test_setting_a_background_renormalizes_the_hologram(self):
        tool = self.tool()
        before = tool.data.copy()
        self.assertTrue(tool.setBackground(50))
        np.testing.assert_allclose(tool.data, tool._raw / 50.)
        self.assertFalse(np.allclose(tool.data, before))

    def test_the_dark_count_is_subtracted_before_dividing(self):
        tool = self.tool(background=100)
        tool.setDark(20.)
        np.testing.assert_allclose(tool.data, (tool._raw - 20.) / 80.)
        self.assertEqual(self.shown(tool), 'background: 100, dark 20')
        # it stays when the background changes
        tool.setBackground(120)
        np.testing.assert_allclose(tool.data, (tool._raw - 20.) / 100.)
        with self.assertRaises(ValueError):
            tool.setDark(-1.)
        self.assertEqual(tool.dark, 20.)

    def test_a_dark_count_at_startup(self):
        tool = LMTool(ALMWidget, str(IMAGE), 100, 20.)
        self.addCleanup(tool.close)
        self.wait(tool)
        np.testing.assert_allclose(tool.data, (tool._raw - 20.) / 80.)

    def test_a_background_of_another_shape_falls_back_to_the_median(self):
        tool = self.tool()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'small.png'
            cv2.imwrite(str(path), np.full((8, 8), 100, dtype=np.uint8))
            with self.assertLogs(level='ERROR'):
                self.assertTrue(tool.setBackground(str(path)))
        self.assertIn('using the median', self.shown(tool))
        np.testing.assert_allclose(tool.data,
                                   Normalizer()(tool._raw))
        # a background of the right shape brings the label back
        self.assertTrue(tool.setBackground(100))
        self.assertEqual(self.shown(tool), 'background: 100')

    def test_the_dialog_sets_the_background(self):
        tool = self.tool()
        with mock.patch('pylorenzmie.lmtool.LMTool.QFileDialog.'
                        'getOpenFileName', return_value=(str(IMAGE), '')):
            tool.readBackground()
        self.assertEqual(tool.normalizer.source, str(IMAGE))

    def test_a_cancelled_dialog_changes_nothing(self):
        tool = self.tool(background=100)
        with mock.patch('pylorenzmie.lmtool.LMTool.QFileDialog.'
                        'getOpenFileName', return_value=('', '')):
            tool.readBackground()
        self.assertEqual(tool.normalizer.reference, 100.)


@unittest.skipUnless(_qt, 'a Qt binding is not available')
class TestModelChoice(ToolTestCase):

    def parse(self, *argv):
        from pylorenzmie.lmtool.arguments import argument_parser
        return argument_parser().parse_args(argv)

    def test_the_defaults(self):
        args = self.parse()
        self.assertFalse(args.distorted)
        self.assertIsNone(args.backend)

    def test_the_flags(self):
        args = self.parse('-d', '-B', 'numpy')
        self.assertTrue(args.distorted)
        self.assertEqual(args.backend, 'numpy')
        args = self.parse('--distorted', '--backend', 'numba')
        self.assertTrue(args.distorted)
        self.assertEqual(args.backend, 'numba')

    def test_b_is_still_the_background(self):
        self.assertEqual(self.parse('-b', '100').background, '100')

    def test_the_dark_count_is_an_option(self):
        self.assertEqual(self.parse().dark, 0.)
        self.assertEqual(self.parse('-D', '29').dark, 29.)
        self.assertEqual(self.parse('--dark', '28.5').dark, 28.5)

    def test_an_unknown_backend_is_refused(self):
        with self.assertRaises(SystemExit):
            with mock.patch('sys.stderr'):
                self.parse('--backend', 'fortran')

    def test_an_unusable_choice_is_a_usage_error(self):
        # reported before a window or an application exists
        from pylorenzmie.lmtool.launch import lmtool
        argv = ['lmtool', '--distorted', '--backend', 'torch']
        with mock.patch('sys.argv', argv), mock.patch('sys.stderr') as err:
            with self.assertRaises(SystemExit) as raised:
                lmtool()
        self.assertEqual(raised.exception.code, 2)
        message = ''.join(call.args[0] for call in err.write.call_args_list)
        self.assertIn('Distorted does not support', message)

    def test_the_panel_class_uses_the_model(self):
        from pylorenzmie.theory import model_class
        model = model_class('numpy', distorted=True)
        panel = ALMWidget.with_model(model)
        self.assertTrue(issubclass(panel, ALMWidget))
        self.assertIs(panel.cls, model)
        self.assertIsNot(ALMWidget.cls, model)

    def test_a_window_with_a_distorted_model(self):
        from pylorenzmie.theory import model_class
        model = model_class('numpy', distorted=True)
        tool = self.tool(ALMWidget.with_model(model))
        self.assertIs(tool.session.cls, model)
        self.assertTrue(hasattr(tool.session.model, 'warp'))
        tool.session.update(dict(spherical=2.))
        tool.estimate()
        self.wait(tool)
        tool.optimize()
        self.wait(tool)
        self.assertIn('spherical', tool.session.properties)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
