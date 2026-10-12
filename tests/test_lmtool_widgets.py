'''Tests of the widgets of lmtool, without a display.

They need a Qt binding (``pip install pylorenzmie[gui]``) and are skipped
without one.  The whole application, LMTool, is not created here.
'''
import os
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

try:
    from pyqtgraph import mkQApp
    from pylorenzmie.lmtool.DoubleSlider import DoubleSlider
    from pylorenzmie.lmtool.DoubleSpinBox import DoubleSpinBox
    from pylorenzmie.lmtool.ParameterWidget import ParameterWidget
    from pylorenzmie.lmtool.LMWidget import LMWidget
    from pylorenzmie.lmtool.ALMWidget import ALMWidget
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

    def test_the_piston_is_held_fixed_until_asked_for(self):
        panel = ALMWidget()
        self.assertIn('piston', panel.fixed)
        self.assertNotIn('spherical', panel.fixed)
        panel.piston.setValue(0.25)
        self.assertAlmostEqual(panel.model.piston, 0.25, delta=0.01)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
