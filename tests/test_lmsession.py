import os
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

try:
    from pyqtgraph import mkQApp
    from pylorenzmie.lmtool.LMSession import LMSession
    from pylorenzmie.lmtool.Preview import Preview
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


@unittest.skipUnless(_qt, 'a Qt binding is not available')
class TestPreview(unittest.TestCase):

    def setUp(self):
        self.session = LMSession(AberratedLorenzMie)
        self.preview = Preview(self.session)
        self.shown = []
        self.preview.updated.connect(
            lambda h, p, n: self.shown.append((h, p, n)))
        self.coordinates = self.session.model.meshgrid((40, 40))[:2]
        self.preview.setCoordinates(self.coordinates)
        self.preview.setRadius(30)

    def process(self):
        mkQApp().processEvents()

    def test_requests_are_computed_once(self):
        self.process()
        before = self.preview.count
        for z in (200., 210., 220., 230.):
            self.session.set('z_p', z)
        self.preview.setRadius(25)
        self.assertEqual(self.preview.count, before)       # not yet
        self.process()
        self.assertEqual(self.preview.count, before + 1)

    def test_nothing_is_computed_without_a_request(self):
        self.process()
        before = self.preview.count
        self.process()
        self.assertEqual(self.preview.count, before)

    def test_the_prediction_has_the_shape_of_the_region_and_the_profile(self):
        self.preview.refresh()
        hologram, profile, noise = self.shown[-1]
        self.assertEqual(hologram.shape, (40 * 40,))
        self.assertEqual(profile.shape, (30,))
        self.assertEqual(noise, self.session.get('noise'))

    def test_the_prediction_follows_the_session(self):
        self.preview.refresh()
        first = self.shown[-1][0]
        self.session.update(dict(z_p=300., a_p=1.2, x_p=20., y_p=20.))
        self.preview.refresh()
        self.assertGreater(abs(self.shown[-1][0] - first).max(), 1e-3)
        expected = self.session.cls(coordinates=self.coordinates)
        expected.properties = self.session.properties
        self.assertTrue((self.shown[-1][0] == expected.hologram()).all())

    def test_the_profile_is_of_a_particle_at_the_origin(self):
        self.session.update(dict(z_p=300., x_p=20., y_p=33.))
        self.preview.refresh()
        moved = self.shown[-1][1]
        self.session.update(dict(x_p=5., y_p=7.))
        self.preview.refresh()
        self.assertTrue((self.shown[-1][1] == moved).all())

    def test_without_a_region_there_is_only_a_profile(self):
        preview = Preview(self.session)
        shown = []
        preview.updated.connect(lambda h, p, n: shown.append((h, p)))
        preview.refresh()
        self.assertIsNone(shown[0][0])
        self.assertEqual(len(shown[0][1]), 100)

    def test_it_does_not_touch_the_model_of_the_session(self):
        # the model that a fit changes in another thread is not the preview's
        model = self.session.model
        coordinates = model.coordinates.copy()
        self.session.set('z_p', 250.)
        self.preview.refresh()
        self.assertEqual(model.coordinates.shape, coordinates.shape)
        self.assertTrue((model.coordinates == coordinates).all())
        self.assertIsNot(self.preview._model, model)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
