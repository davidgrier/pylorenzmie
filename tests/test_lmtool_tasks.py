import os
import threading
import time
import unittest

import numpy as np

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

try:
    from pyqtgraph import mkQApp
    from pylorenzmie.lmtool.tasks import (Cancelled, Snapshot, TaskRunner,
                                          run_estimate, run_fit)
    from pylorenzmie.theory import AberratedLorenzMie, LorenzMie
    _qt = True
except Exception:       # no Qt binding, or no usable platform plugin
    _qt = False


def setUpModule():
    if _qt:
        mkQApp()


def snapshot(**kwargs):
    '''A noisy hologram of a known particle, and a start that is off.'''
    shape = (100, 100)
    coordinates = LorenzMie.meshgrid(shape, flatten=False)
    truth = LorenzMie(coordinates=LorenzMie.meshgrid(shape))
    truth.particle.r_p = [50., 50., 300.]
    truth.particle.a_p, truth.particle.n_p = 1.0, 1.45
    noise = np.random.default_rng(0).normal(0., 0.03, shape)
    data = truth.hologram().reshape(shape) + noise
    start = AberratedLorenzMie().properties
    start.update(x_p=48., y_p=52., z_p=330., a_p=0.9, n_p=1.4, noise=0.03)
    start.update(kwargs.pop('properties', {}))
    fixed = kwargs.pop('fixed', ['wavelength', 'magnification', 'n_m',
                                 'k_p', 'noise', 'defocus', 'spherical',
                                 'coma', 'astigmatism', 'distortion', 'x_c',
                                 'y_c', 'piston'])
    return Snapshot.take(AberratedLorenzMie, start, data, coordinates, fixed,
                         **kwargs)


@unittest.skipUnless(_qt, 'a Qt binding is not available')
class TestSnapshot(unittest.TestCase):

    def test_a_snapshot_has_copies(self):
        data = np.ones((4, 4))
        coordinates = np.zeros((2, 4, 4))
        properties = dict(z_p=1.)
        snap = Snapshot.take(LorenzMie, properties, data, coordinates)
        data[:] = 5.
        coordinates[:] = 5.
        properties['z_p'] = 5.
        self.assertEqual(snap.data.max(), 1.)
        self.assertEqual(snap.coordinates.max(), 0.)
        self.assertEqual(snap.properties['z_p'], 1.)

    def test_a_snapshot_is_frozen(self):
        snap = snapshot()
        with self.assertRaises(AttributeError):
            snap.fraction = 0.5


@unittest.skipUnless(_qt, 'a Qt binding is not available')
class TestTasks(unittest.TestCase):

    def test_a_fit_finds_the_particle(self):
        fit = run_fit(snapshot(), threading.Event())
        self.assertAlmostEqual(fit.properties['z_p'], 300., delta=15.)
        self.assertAlmostEqual(fit.properties['a_p'], 1.0, delta=0.1)
        self.assertAlmostEqual(fit.properties['n_p'], 1.45, delta=0.03)
        self.assertTrue(fit.result['success'])

    def test_a_fit_holds_the_fixed_parameters(self):
        snap = snapshot(properties=dict(spherical=1.5, piston=0.2))
        fit = run_fit(snap, threading.Event())
        self.assertEqual(fit.properties['spherical'], 1.5)
        self.assertEqual(fit.properties['piston'], 0.2)
        self.assertNotIn('dspherical', fit.result)

    def test_a_fit_can_free_a_parameter(self):
        fixed = [name for name in snapshot().fixed if name != 'spherical']
        fit = run_fit(snapshot(fixed=fixed), threading.Event())
        self.assertIn('dspherical', fit.result)

    def test_the_metadata_has_the_fixed_parameters(self):
        fit = run_fit(snapshot(), threading.Event())
        self.assertIn('wavelength', fit.metadata)
        self.assertEqual(fit.metadata['noise'], 0.03)

    def test_a_fit_leaves_the_snapshot_alone(self):
        snap = snapshot()
        data, properties = snap.data.copy(), dict(snap.properties)
        run_fit(snap, threading.Event())
        self.assertTrue((snap.data == data).all())
        self.assertEqual(snap.properties, properties)

    def test_a_fit_that_is_cancelled_stops(self):
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(Cancelled):
            run_fit(snapshot(), cancel)
        with self.assertRaises(Cancelled):
            run_estimate(snapshot(), cancel)

    def test_a_fit_stops_in_the_middle(self):
        cancel = threading.Event()
        snap = snapshot()
        calls = []

        class Counting(AberratedLorenzMie):
            def hologram(self, *args, **kwargs):
                calls.append(1)
                if len(calls) == 5:
                    cancel.set()
                return super().hologram(*args, **kwargs)

        snap = Snapshot.take(Counting, snap.properties, snap.data,
                             snap.coordinates, snap.fixed)
        with self.assertRaises(Cancelled):
            run_fit(snap, cancel)
        self.assertLess(len(calls), 20)

    def test_an_estimate_returns_the_particle(self):
        result = run_estimate(snapshot(), threading.Event())
        for name in ('z_p', 'a_p', 'n_p'):
            self.assertIn(name, result)
        self.assertGreater(result['z_p'], 10.)

    def test_an_estimate_keeps_the_fixed_parameters(self):
        fixed = snapshot()
        fixed = Snapshot.take(fixed.cls, fixed.properties, fixed.data,
                              fixed.coordinates, [*fixed.fixed, 'n_p'])
        result = run_estimate(fixed, threading.Event())
        self.assertEqual(result['n_p'], fixed.properties['n_p'])
        self.assertNotEqual(result['a_p'], fixed.properties['a_p'])

    def test_an_estimate_with_everything_fixed_changes_nothing(self):
        fixed = snapshot()
        fixed = Snapshot.take(fixed.cls, fixed.properties, fixed.data,
                              fixed.coordinates,
                              [*fixed.fixed, 'z_p', 'a_p', 'n_p'])
        result = run_estimate(fixed, threading.Event())
        for name in ('z_p', 'a_p', 'n_p'):
            self.assertEqual(result[name], fixed.properties[name])


@unittest.skipUnless(_qt, 'a Qt binding is not available')
class TestTaskRunner(unittest.TestCase):

    def setUp(self):
        self.runner = TaskRunner()
        self.addCleanup(self.runner.shutdown)
        self.events = []
        self.threads = []
        note = self.events.append
        self.runner.started.connect(lambda n: note(('started', n)))
        self.runner.finished.connect(self.finished)
        self.runner.failed.connect(lambda n, m: note(('failed', n, m)))
        self.runner.cancelled.connect(lambda n: note(('cancelled', n)))

    def finished(self, name, result):
        self.threads.append(threading.current_thread())
        self.events.append(('finished', name, result))

    def wait(self, seconds=20):
        end = time.time() + seconds
        while self.runner.busy and time.time() < end:
            mkQApp().processEvents()
            time.sleep(0.005)
        mkQApp().processEvents()

    def test_a_result_arrives_on_the_interface_thread(self):
        self.assertTrue(self.runner.start('sum', lambda s, c: s + 1, 41))
        self.assertTrue(self.runner.busy)
        self.wait()
        self.assertEqual(self.events, [('started', 'sum'),
                                       ('finished', 'sum', 42)])
        self.assertIs(self.threads[0], threading.main_thread())
        self.assertFalse(self.runner.busy)

    def test_one_task_at_a_time(self):
        release = threading.Event()
        self.runner.start('slow', lambda s, c: release.wait(10), None)
        self.assertFalse(self.runner.start('other', lambda s, c: 0, None))
        release.set()
        self.wait()
        self.assertTrue(self.runner.start('again', lambda s, c: 1, None))
        self.wait()
        self.assertEqual([e[1] for e in self.events if e[0] == 'finished'],
                         ['slow', 'again'])

    def test_cancel_stops_the_task(self):
        def slow(snapshot, cancel):
            cancel.wait(10)
            raise Cancelled
        self.runner.start('slow', slow, None)
        self.runner.cancel()
        self.wait()
        self.assertEqual(self.events[-1], ('cancelled', 'slow'))
        self.assertFalse(self.runner.busy)

    def test_a_failure_is_reported_with_its_kind(self):
        def broken(snapshot, cancel):
            raise ValueError('no good')
        with self.assertLogs('pylorenzmie.lmtool.tasks', level='ERROR'):
            self.runner.start('broken', broken, None)
            self.wait()
        self.assertEqual(self.events[-1],
                         ('failed', 'broken', 'ValueError: no good'))

    def test_a_cancel_does_not_stop_the_next_task(self):
        self.runner.start('first', lambda s, c: c.wait(10) and 0, None)
        self.runner.cancel()
        self.wait()
        self.runner.start('second', lambda s, c: 7, None)
        self.wait()
        self.assertEqual(self.events[-1], ('finished', 'second', 7))


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
