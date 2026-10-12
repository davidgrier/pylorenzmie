'''Tests of the widgets of lmtool, without a display.

They need a Qt binding (``pip install pylorenzmie[gui]``) and are skipped
without one.  :class:`LMTool` is created with a file name: without one it
opens a modal file dialog and waits for a person.
'''
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

try:
    from pyqtgraph import mkQApp
    from pylorenzmie.lmtool.DoubleSlider import DoubleSlider
    from pylorenzmie.lmtool.DoubleSpinBox import DoubleSpinBox
    from pylorenzmie.lmtool.ParameterWidget import ParameterWidget
    from pylorenzmie.lmtool.LMWidget import LMWidget
    from pylorenzmie.lmtool.ALMWidget import ALMWidget
    from pylorenzmie.lmtool.FitWidget import FitWidget
    from pylorenzmie.lmtool.LMTool import LMTool
    _qt = True
except Exception:       # no Qt binding, or no usable platform plugin
    _qt = False


def setUpModule():
    if _qt:
        mkQApp()


@unittest.skipUnless(_qt, 'a Qt binding is not available')
class TestDoubleSlider(unittest.TestCase):

    def setUp(self):
        self.slider = DoubleSlider()

    def test_default_range(self):
        self.assertEqual(self.slider.minimum(), 0.)
        self.assertEqual(self.slider.maximum(), 100.)

    def test_value_round_trip(self):
        self.slider.setValue(25.)
        self.assertAlmostEqual(self.slider.value(), 25., delta=0.02)

    def test_value_is_limited_to_the_range(self):
        self.slider.setValue(500.)
        self.assertAlmostEqual(self.slider.value(), 100., delta=0.02)
        self.slider.setValue(-5.)
        self.assertAlmostEqual(self.slider.value(), 0., delta=0.02)

    def test_range(self):
        self.slider.setRange(-2., 2.)
        self.assertEqual((self.slider.minimum(), self.slider.maximum()),
                         (-2., 2.))
        self.slider.setValue(1.)
        self.assertAlmostEqual(self.slider.value(), 1., delta=0.001)

    def test_an_impossible_limit_is_refused_and_logged(self):
        with self.assertLogs('pylorenzmie.lmtool.DoubleSlider',
                             level='WARNING'):
            self.slider.setMinimum(200.)
        self.assertEqual(self.slider.minimum(), 0.)

    def test_signal_carries_a_float(self):
        seen = []
        self.slider.valueChanged[float].connect(seen.append)
        self.slider.setValue(40.)
        self.assertEqual(len(seen), 1)
        self.assertAlmostEqual(seen[0], 40., delta=0.02)


@unittest.skipUnless(_qt, 'a Qt binding is not available')
class TestDoubleSpinBox(unittest.TestCase):

    def test_a_step_that_changes_the_value_is_reported(self):
        box = DoubleSpinBox()
        box.setRange(0., 10.)
        box.setSingleStep(1.)
        seen = []
        box.buttonClicked.connect(seen.append)
        box.stepBy(1)
        self.assertEqual(seen, [1.])

    def test_a_step_at_the_limit_is_not_reported(self):
        box = DoubleSpinBox()
        box.setRange(0., 10.)
        box.setValue(10.)
        seen = []
        box.buttonClicked.connect(seen.append)
        box.stepBy(1)
        self.assertEqual(seen, [])


@unittest.skipUnless(_qt, 'a Qt binding is not available')
class TestParameterWidget(unittest.TestCase):

    def test_settings_round_trip(self):
        widget = ParameterWidget()
        settings = dict(text='radius', range=[0.5, 5.], value=1.5, step=0.25,
                        fixed=True, decimals=3)
        widget.setSettings(settings)
        out = widget.settings()
        for key, value in settings.items():
            with self.subTest(key=key):
                if key == 'value':
                    self.assertAlmostEqual(out[key], value, delta=0.01)
                else:
                    self.assertEqual(out[key], value)

    def test_prefix_and_suffix_are_reported_only_when_set(self):
        widget = ParameterWidget()
        self.assertNotIn('suffix', widget.settings())
        widget.setSettings(dict(suffix=' um'))
        self.assertEqual(widget.settings()['suffix'], ' um')

    def test_fixed(self):
        widget = ParameterWidget()
        self.assertFalse(widget.fixed())
        widget.setFixed(True)
        self.assertTrue(widget.fixed())

    def test_changing_the_value_emits_valueChanged(self):
        widget = ParameterWidget()
        widget.setRange([0., 10.])
        seen = []
        widget.valueChanged.connect(seen.append)
        widget.setValue(4.)
        self.assertEqual(len(seen), 1)
        self.assertAlmostEqual(seen[0], 4., delta=0.01)


@unittest.skipUnless(_qt, 'a Qt binding is not available')
class TestControlPanels(unittest.TestCase):

    def test_the_panel_controls_the_model(self):
        panel = LMWidget()
        for name in ('wavelength', 'magnification', 'n_m', 'a_p', 'n_p',
                     'x_p', 'y_p', 'z_p'):
            self.assertIn(name, panel.properties)
        seen = []
        panel.propertyChanged.connect(lambda n, v: seen.append((n, v)))
        panel.a_p.setValue(1.25)
        self.assertEqual(seen[0][0], 'a_p')
        self.assertAlmostEqual(panel.model.particle.a_p, 1.25, delta=0.01)

    def test_the_aberrated_panel_has_the_aberration_controls(self):
        panel = ALMWidget()
        self.assertIn('spherical', panel.properties)
        self.assertNotIn('numerical_aperture', panel.properties)
        self.assertIn('spherical', panel.model.properties)

    def test_the_second_row_of_the_instrument_group(self):
        panel = ALMWidget()
        layout = panel.groupInstrument.layout()
        row = {layout.getItemPosition(i)[1]: layout.itemAt(i).widget()
               for i in range(layout.count())
               if layout.getItemPosition(i)[0] == 1}
        self.assertEqual([row[c].objectName() for c in sorted(row)],
                         ['noise', 'piston', 'spherical'])

    def test_properties_without_a_control_are_fixed(self):
        # a fit must not free parameters that the user cannot see: defocus
        # is nearly degenerate with the height, and the aberration center
        # means nothing without coma, astigmatism or distortion
        from pylorenzmie.analysis import Optimizer
        panel = ALMWidget()
        for name in ('defocus', 'coma', 'astigmatism', 'distortion',
                     'x_c', 'y_c'):
            self.assertIn(name, panel.fixed)
        optimizer = Optimizer(model=panel.model)
        optimizer.fixed = panel.fixed
        self.assertEqual(optimizer.variables,
                         ['x_p', 'y_p', 'z_p', 'a_p', 'n_p'])

    def test_piston_and_spherical_are_held_fixed_until_asked_for(self):
        panel = ALMWidget()
        self.assertIn('piston', panel.fixed)
        self.assertIn('spherical', panel.fixed)
        panel.piston.setValue(0.25)
        self.assertAlmostEqual(panel.model.piston, 0.25, delta=0.01)


@unittest.skipUnless(_qt, 'a Qt binding is not available')
class TestSavingResults(unittest.TestCase):

    def setUp(self):
        self.widget = FitWidget()
        self.widget.result = pd.Series({'x_p': 1., 'z_p': 100.})
        self.saved, self.failed = [], []
        self.widget.resultSaved.connect(self.saved.append)
        self.widget.saveFailed.connect(self.failed.append)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def path(self, name):
        return str(Path(self.directory.name) / name)

    def test_nothing_to_save_is_reported(self):
        self.widget.result = None
        self.widget.saveResult(self.path('a.h5'))
        self.assertEqual(self.saved, [])
        self.assertIn('no result', self.failed[0])

    def test_json_is_saved_and_reported(self):
        filename = self.path('result.json')
        self.widget.saveJson(filename)
        self.assertEqual(self.saved, [filename])
        self.assertEqual(json.loads(Path(filename).read_text())['z_p'], 100.)

    def test_without_pytables_the_result_goes_to_json(self):
        with mock.patch.object(self.widget, '_saveHdf',
                               side_effect=ImportError):
            self.widget.saveResult(self.path('result.h5'))
        self.assertEqual(self.saved, [self.path('result.json')])
        self.assertTrue(Path(self.saved[0]).exists())

    def test_hdf5_is_saved_when_pytables_is_available(self):
        try:
            import tables  # noqa: F401
        except ImportError:
            self.skipTest('PyTables is not installed')
        filename = self.path('result.h5')
        self.widget.saveResult(filename)
        self.assertEqual(self.saved, [filename])
        self.assertEqual(pd.read_hdf(filename, key='result')['z_p'], 100.)
        self.assertIn('datafile', pd.read_hdf(filename, key='metadata'))

    def test_a_failure_is_reported_not_raised(self):
        with mock.patch.object(self.widget, '_saveHdf',
                               side_effect=OSError('disk full')):
            self.widget.saveResult(self.path('result.h5'))
        self.assertEqual(self.saved, [])
        self.assertIn('disk full', self.failed[0])

    def test_the_default_directory_is_created(self):
        with mock.patch.object(Path, 'expanduser',
                               return_value=Path(self.directory.name) /
                               'new' / 'lmtool'):
            path = self.widget._output_path()
        self.assertTrue(Path(path).parent.is_dir())


@unittest.skipUnless(_qt, 'a Qt binding is not available')
class TestLMTool(unittest.TestCase):

    IMAGE = Path(__file__).parents[1] / 'docs' / 'tutorials' / 'crop.png'

    def test_a_session(self):
        tool = LMTool(ALMWidget, str(self.IMAGE))
        self.addCleanup(tool.close)
        self.assertEqual(tool._data.shape, (390, 390))
        # the ROI controls the position of the model
        tool.imageWidget.x_p, tool.imageWidget.y_p = 195.5, 193.2
        properties = tool.controls.properties
        self.assertAlmostEqual(properties['x_p'], 195.5, delta=0.1)
        self.assertAlmostEqual(properties['y_p'], 193.2, delta=0.1)
        # a fit changes the shown parameters and not the hidden ones
        tool.fitWidget.optimizer.model.properties = dict(defocus=0.)
        result = tool.fitWidget.optimize(*(tool.crop()[0::2]))
        for name in ('defocus', 'coma', 'astigmatism', 'distortion',
                     'x_c', 'y_c', 'spherical', 'piston'):
            self.assertEqual(tool.fitWidget.model.properties[name], 0.)
        self.assertTrue(result['success'])

    def test_the_roi_starts_on_the_particle(self):
        tool = LMTool(ALMWidget, str(self.IMAGE))
        self.addCleanup(tool.close)
        properties = tool.controls.properties
        self.assertAlmostEqual(properties['x_p'], 195.5, delta=2.)
        self.assertAlmostEqual(properties['y_p'], 193.2, delta=2.)
        self.assertGreaterEqual(tool.imageWidget.radius, 30)
        self.assertIn('Only one particle', tool.statusBar().currentMessage())

    def test_next_particle_cycles_through_the_candidates(self):
        import pandas as pd
        tool = LMTool(ALMWidget, str(self.IMAGE))
        self.addCleanup(tool.close)
        tool._candidates = pd.DataFrame(
            [dict(x_p=100., y_p=120., bbox=((50, 70), 100, 100)),
             dict(x_p=250., y_p=200., bbox=((190, 140), 120, 120))])
        tool._candidate = -1
        tool.nextParticle()
        self.assertAlmostEqual(tool.controls.properties['x_p'], 100., delta=1.)
        tool.nextParticle()
        self.assertAlmostEqual(tool.controls.properties['x_p'], 250., delta=1.)
        self.assertEqual(tool.imageWidget.radius, 60)
        tool.nextParticle()                       # and around again
        self.assertAlmostEqual(tool.controls.properties['y_p'], 120., delta=1.)

    def test_one_particle_is_said_to_be_the_only_one(self):
        tool = LMTool(ALMWidget, str(self.IMAGE))
        self.addCleanup(tool.close)
        tool.nextParticle()
        self.assertIn('Only one particle', tool.statusBar().currentMessage())

    def test_many_candidates_are_limited_and_ranked(self):
        import numpy as np
        tool = LMTool(ALMWidget, str(self.IMAGE))
        self.addCleanup(tool.close)
        noise = np.random.default_rng(1).normal(1., 0.03, (400, 400))
        tool.data = noise
        tool.findParticles()
        found = tool._candidates
        self.assertLessEqual(len(found), tool.MAX_CANDIDATES)
        self.assertGreater(found.attrs['nfound'], len(found))
        self.assertEqual(list(found['mass']),
                         sorted(found['mass'], reverse=True))
        self.assertIn('noisy', tool.statusBar().currentMessage())

    def test_no_particles_is_reported(self):
        import numpy as np
        tool = LMTool(ALMWidget, str(self.IMAGE))
        self.addCleanup(tool.close)
        tool.data = np.ones((200, 200))
        tool.findParticles()
        self.assertIn('No particles', tool.statusBar().currentMessage())

    def test_the_status_bar_reports_a_saved_result(self):
        tool = LMTool(ALMWidget, str(self.IMAGE))
        self.addCleanup(tool.close)
        tool.fitWidget.saveFailed.emit('Could not save the result: test')
        self.assertIn('Could not save', tool.statusBar().currentMessage())


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
