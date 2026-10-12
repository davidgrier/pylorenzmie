import os
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

try:
    from pyqtgraph import mkQApp
    from pylorenzmie.lmtool.LMSession import LMSession
    from pylorenzmie.theory import AberratedLorenzMie
    _qt = True
except Exception:       # no Qt binding, or no usable platform plugin
    _qt = False


def setUpModule():
    if _qt:
        mkQApp()


@unittest.skipUnless(_qt, 'a Qt binding is not available')
class TestLMSession(unittest.TestCase):

    def setUp(self):
        self.session = LMSession(AberratedLorenzMie)
        self.announced = []
        self.session.parametersChanged.connect(self.announced.append)

    def test_the_session_has_every_parameter_of_the_model(self):
        self.assertEqual(self.session.properties.keys(),
                         self.session.model.properties.keys())
        self.assertIn('defocus', self.session.properties)

    def test_set_changes_the_model_and_announces_once(self):
        self.session.update(dict(z_p=300., a_p=1.1, n_p=1.6))
        self.assertEqual(len(self.announced), 1)
        self.assertEqual(sorted(self.announced[0]), ['a_p', 'n_p', 'z_p'])
        self.assertEqual(self.session.model.particle.z_p, 300.)
        self.assertEqual(self.session.get('a_p'), 1.1)

    def test_setting_the_same_value_announces_nothing(self):
        self.session.set('z_p', 300.)
        self.session.set('z_p', 300.)
        self.session.set('z_p', 300. + 1e-12)
        self.assertEqual(len(self.announced), 1)

    def test_only_what_changed_is_announced(self):
        self.session.set('z_p', 300.)
        self.session.update(dict(z_p=300., a_p=1.2))
        self.assertEqual(self.announced[-1], ['a_p'])

    def test_unknown_names_are_ignored(self):
        changed = self.session.update(dict(not_a_parameter=1.))
        self.assertEqual(changed, [])
        self.assertEqual(self.announced, [])

    def test_properties_is_a_copy(self):
        self.session.properties['z_p'] = 1.
        self.assertNotEqual(self.session.get('z_p'), 1.)

    def test_adopt_announces_what_the_model_changed(self):
        # a fit changes the model without telling the session
        self.session.model.properties = dict(z_p=420., n_p=1.6)
        self.assertEqual(self.session.get('z_p'), 100.)
        changed = self.session.adopt()
        self.assertEqual(sorted(changed), ['n_p', 'z_p'])
        self.assertEqual(self.session.get('z_p'), 420.)
        self.assertEqual(len(self.announced), 1)
        self.assertEqual(self.session.adopt(), [])

    def test_restore_undoes_what_the_model_changed(self):
        self.session.set('z_p', 300.)
        self.session.model.properties = dict(z_p=999., a_p=3.)
        self.session.restore()
        self.assertEqual(self.session.model.particle.z_p, 300.)
        self.assertNotEqual(self.session.model.particle.a_p, 3.)
        self.assertEqual(len(self.announced), 1)

    def test_parameters_that_are_not_shown_are_fixed(self):
        fixed = []
        self.session.fixedChanged.connect(fixed.append)
        self.assertEqual(self.session.fixed, [])      # nothing is hidden yet
        self.session.setShown(['x_p', 'y_p', 'z_p', 'a_p', 'n_p'])
        self.assertIn('defocus', self.session.fixed)
        self.assertIn('x_c', self.session.fixed)
        self.assertNotIn('z_p', self.session.fixed)
        self.assertTrue(self.session.isFixed('coma'))
        self.assertEqual(len(fixed), 1)
        self.assertNotIn('z_p', self.session.hidden)

    def test_a_parameter_that_is_shown_can_be_fixed_and_freed(self):
        self.session.setShown(['x_p', 'z_p'])
        fixed = []
        self.session.fixedChanged.connect(fixed.append)
        self.session.setFixed('z_p')
        self.assertIn('z_p', self.session.fixed)
        self.session.setFixed('z_p')                  # no change, no signal
        self.assertEqual(len(fixed), 1)
        self.session.setFixed('z_p', False)
        self.assertNotIn('z_p', self.session.fixed)
        self.assertEqual(len(fixed), 2)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
