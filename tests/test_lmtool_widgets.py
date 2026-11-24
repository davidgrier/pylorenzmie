'''Tests of the widgets of lmtool, without a display.

They need a Qt binding (``pip install pylorenzmie[gui]``) and are skipped
without one.  The application window is tested in ``test_lmtool.py``.
'''
import os
import unittest

import numpy as np

from pylorenzmie.theory import LorenzMie

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

try:
    import qtpy     # chooses the Qt binding (QT_API) before pyqtgraph can
    from pyqtgraph import mkQApp
    from pylorenzmie.lmtool.DoubleSlider import DoubleSlider
    from pylorenzmie.lmtool.DoubleSpinBox import DoubleSpinBox
    from pylorenzmie.lmtool.ParameterWidget import ParameterWidget
    from pylorenzmie.lmtool.LMWidget import LMWidget
    from pylorenzmie.lmtool.ALMWidget import ALMWidget
    from pylorenzmie.lmtool.OptimizerWidget import OptimizerWidget
    from pylorenzmie.lmtool.ProfileWidget import ProfileWidget
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
        settings = dict(text='radius',
                        range=[0.5, 5.],
                        value=1.5,
                        step=0.25,
                        fixed=True,
                        decimals=3)
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
class TestOptimizerWidget(unittest.TestCase):

    def setUp(self):
        self.widget = OptimizerWidget()
        self.emitted = []
        self.widget.settingChanged.connect(
            lambda name, value: self.emitted.append((name, value)))

    def test_settings_are_those_of_least_squares(self):
        self.assertEqual(
            set(self.widget.settings),
            {'fraction', 'method', 'loss', 'ftol', 'xtol', 'gtol',
             'diff_step'},
        )

    def test_levenberg_marquardt_is_the_default(self):
        self.assertEqual(self.widget.settings['method'], 'lm')

    def test_levenberg_marquardt_requires_the_linear_loss(self):
        loss = self.widget.params.child('loss')
        self.assertEqual(loss.value(), 'linear')
        self.assertFalse(loss.opts['enabled'])

    def test_another_method_releases_the_loss(self):
        self.widget.settings = {'method': 'trf'}
        self.assertTrue(self.widget.params.child('loss').opts['enabled'])
        self.widget.settings = {'loss': 'huber'}
        self.assertEqual(self.widget.settings['loss'], 'huber')

    def test_going_back_to_lm_restores_the_linear_loss(self):
        self.widget.settings = {'method': 'trf', 'loss': 'huber'}
        self.widget.settings = {'method': 'lm'}
        self.assertEqual(self.widget.settings['loss'], 'linear')

    def test_a_change_is_announced(self):
        self.widget.settings = {'fraction': 0.5}
        self.assertIn(('fraction', 0.5), self.emitted)

    def test_an_unknown_setting_is_ignored(self):
        before = self.widget.settings
        self.widget.settings = {'no_such_setting': 1}
        self.assertEqual(self.widget.settings, before)


@unittest.skipUnless(_qt, 'a Qt binding is not available')
class TestProfileWidget(unittest.TestCase):

    def setUp(self):
        self.widget = ProfileWidget(radius=50)

    def curve(self, item):
        return item.getData()[1]

    def test_the_model_is_evaluated_along_a_radius(self):
        self.assertEqual(self.widget.model.coordinates.shape[1], 50)
        self.assertEqual(len(self.curve(self.widget.theory)), 50)

    def test_the_model_is_centered_at_the_origin(self):
        particle = self.widget.model.single()
        self.assertEqual(particle.x_p, 0.)
        self.assertEqual(particle.y_p, 0.)

    def test_a_model_is_moved_to_the_origin(self):
        model = LorenzMie()
        model.single().r_p = [50., 60., 200.]
        self.widget.model = model
        self.assertEqual(model.single().x_p, 0.)
        self.assertEqual(model.single().y_p, 0.)
        self.assertEqual(model.single().z_p, 200.)

    def test_data_are_plotted_with_their_standard_deviation(self):
        profile = np.linspace(1., 2., 80)
        stdev = np.full(80, 0.1)
        self.widget.data = (profile, stdev)
        np.testing.assert_allclose(self.curve(self.widget.experiment),
                                   profile[:50])
        np.testing.assert_allclose(self.curve(self.widget.upperLimit),
                                   profile[:50] + 0.1)
        np.testing.assert_allclose(self.curve(self.widget.lowerLimit),
                                   profile[:50] - 0.1)

    def test_a_short_profile_limits_the_radius_that_is_plotted(self):
        self.widget.data = (np.ones(20), np.zeros(20))
        self.assertEqual(len(self.curve(self.widget.experiment)), 20)

    def test_without_data_nothing_is_plotted(self):
        self.widget.plotData()
        self.assertEqual(len(self.widget.experiment.getData()[0]), 0)

    def test_a_prediction_can_be_shown_without_a_model(self):
        self.widget.model = None
        self.widget.showTheory(np.arange(10.))
        np.testing.assert_array_equal(self.curve(self.widget.theory),
                                      np.arange(10.))

    def test_the_position_in_properties_is_ignored(self):
        before = self.widget.model.properties['z_p']
        self.widget.properties = {'x_p': 99., 'y_p': 99.}
        self.assertEqual(self.widget.model.single().x_p, 0.)
        self.assertEqual(self.widget.model.properties['z_p'], before)

    def test_properties_change_the_prediction(self):
        old = self.curve(self.widget.theory).copy()
        self.widget.properties = {'z_p': 200., 'a_p': 1.0}
        self.assertFalse(np.allclose(self.curve(self.widget.theory), old))


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

    def test_control_finds_a_control_by_name(self):
        panel = LMWidget()
        self.assertIs(panel.control('a_p'), panel.a_p)
        self.assertIsNone(panel.control('noise'))
        self.assertIsNotNone(ALMWidget().control('noise'))

    def test_a_config_names_only_controls(self):
        panel = LMWidget()
        panel.setConfig(dict(model=dict(value=1.), a_p=dict(value=1.5)))
        self.assertEqual([c.objectName() for c in panel.controls], ['a_p'])

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
class TestPanelAndSession(unittest.TestCase):
    '''The panel is a view of the session.'''

    def setUp(self):
        self.panel = ALMWidget()
        self.session = self.panel.session

    def test_editing_a_control_sets_the_parameter(self):
        self.panel.a_p.setValue(1.25)
        self.assertAlmostEqual(self.session.get('a_p'), 1.25, delta=0.01)
        self.assertAlmostEqual(self.session.model.particle.a_p,
                               1.25,
                               delta=0.01)

    def test_a_change_in_the_session_moves_the_control(self):
        self.session.update(dict(z_p=321., n_p=1.6))
        self.assertAlmostEqual(self.panel.z_p.value(), 321., delta=0.5)
        self.assertAlmostEqual(self.panel.n_p.value(), 1.6, delta=0.01)

    def test_the_panel_does_not_echo_what_the_session_announced(self):
        seen = []
        self.panel.propertyChanged.connect(lambda n, v: seen.append(n))
        self.session.update(dict(z_p=321.))
        self.assertEqual(seen, [])
        self.panel.z_p.setValue(250.)
        self.assertEqual(seen, ['z_p'])

    def test_a_value_the_session_holds_exactly_is_not_rounded_by_the_control(
            self):
        self.session.update(dict(noise=0.034552))
        self.assertEqual(self.session.get('noise'), 0.034552)
        self.assertEqual(self.panel.properties['noise'], 0.034552)

    def test_the_fixed_checkbox_and_the_session_agree(self):
        self.assertTrue(self.session.isFixed('piston'))   # from the config
        self.panel.z_p.setFixed(True)
        self.assertIn('z_p', self.session.fixed)
        self.session.setFixed('z_p', False)
        self.assertFalse(self.panel.z_p.fixed())

    def test_two_panels_can_share_a_session(self):
        other = ALMWidget(session=self.session)
        self.panel.z_p.setValue(222.)
        self.assertAlmostEqual(other.z_p.value(), 222., delta=0.5)
        self.assertIs(other.model, self.panel.model)


@unittest.skipUnless(_qt, 'a Qt binding is not available')
class TestParameterSpecifications(unittest.TestCase):
    '''The specifications, the layouts and the models agree.'''

    def names(self, panel):
        return {w.objectName() for w in panel.findChildren(ParameterWidget)}

    def test_every_control_has_a_specification(self):
        from pylorenzmie.lmtool.parameters import PARAMETERS
        specified = {spec.name for spec in PARAMETERS}
        for cls in (LMWidget, ALMWidget):
            with self.subTest(cls.__name__):
                self.assertEqual(self.names(cls()) - specified, set())

    def test_every_control_is_a_property_of_its_model(self):
        for cls in (LMWidget, ALMWidget):
            with self.subTest(cls.__name__):
                panel = cls()
                self.assertEqual(self.names(panel) -
                                 set(panel.model.properties), set())

    def test_the_hidden_parameters_of_the_aberrated_panel(self):
        panel = ALMWidget()
        self.assertEqual(sorted(panel.session.hidden),
                         ['astigmatism', 'coma', 'defocus', 'distortion',
                          'x_c', 'y_c'])

    def test_the_specifications_are_valid_and_unique(self):
        from pylorenzmie.lmtool.parameters import PARAMETERS, ParameterSpec
        names = [spec.name for spec in PARAMETERS]
        self.assertEqual(len(names), len(set(names)))
        with self.assertRaises(ValueError):
            ParameterSpec('a', 'a', (1., 0.), 0.5)
        with self.assertRaises(ValueError):
            ParameterSpec('a', 'a', (0., 1.), 2.)
        with self.assertRaises(ValueError):
            ParameterSpec('a', 'a', (0., 1.), 0.5, step=0.)

    def test_the_controls_are_configured_from_the_specifications(self):
        from pylorenzmie.lmtool.parameters import specification
        panel = ALMWidget()
        for name in ('wavelength', 'a_p', 'noise', 'piston', 'z_p'):
            spec, control = specification(name), getattr(panel, name)
            with self.subTest(name):
                self.assertEqual(control.range(), list(spec.range))
                self.assertEqual(control.decimals(), spec.decimals)
                self.assertEqual(control.fixed(), spec.fixed)
                self.assertEqual(control.suffix(), spec.suffix)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
