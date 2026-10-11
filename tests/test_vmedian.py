import unittest

import numpy as np

from pylorenzmie.utilities import VMedian


def frames(values, shape=(4, 5)):
    return [np.full(shape, v, dtype=np.uint8) for v in values]


class TestVMedian(unittest.TestCase):

    def test_not_initialized_before_three_frames(self):
        v = VMedian()
        for frame in frames([10, 20]):
            v.add(frame)
            self.assertFalse(v.initialized)
        v.add(frames([30])[0])
        self.assertTrue(v.initialized)

    def test_the_median_of_three_frames(self):
        v = VMedian()
        for frame in frames([10, 90, 20]):
            v.add(frame)
        np.testing.assert_array_equal(v.get(), np.full((4, 5), 20))

    def test_a_moving_object_is_removed(self):
        # a bright pixel that is in a different place in each frame
        stack = []
        for k in range(3):
            frame = np.full((6, 6), 50, dtype=np.uint8)
            frame[k, k] = 255
            stack.append(frame)
        v = VMedian()
        for frame in stack:
            v.add(frame)
        np.testing.assert_array_equal(v.get(), np.full((6, 6), 50))

    def test_filter_adds_and_returns_the_estimate(self):
        v = VMedian()
        out = None
        for frame in frames([1, 5, 3]):
            out = v.filter(frame)
        self.assertEqual(out.shape, (4, 5))
        self.assertTrue(np.all(out == 3))

    def test_get_can_return_the_flat_buffer(self):
        v = VMedian()
        for frame in frames([1, 5, 3]):
            v.add(frame)
        self.assertEqual(v.get(reshape=False).shape, (20,))

    def test_cycled_is_true_on_the_third_frame_only(self):
        v = VMedian()
        flags = []
        for frame in frames([1, 2, 3, 4, 5, 6]):
            v.add(frame)
            flags.append(v.cycled)
        self.assertEqual(flags, [False, False, True, False, False, True])

    def test_order_one_needs_nine_frames(self):
        v = VMedian(order=1)
        state = []
        for frame in frames(range(1, 10)):
            v.add(frame)
            state.append(v.initialized)
        self.assertFalse(any(state[:8]))
        self.assertTrue(state[8])
        # the median of the medians of (1, 2, 3), (4, 5, 6), (7, 8, 9)
        self.assertTrue(np.all(v.get() == 5))

    def test_reset(self):
        v = VMedian(order=1)
        for frame in frames(range(1, 10)):
            v.add(frame)
        v.reset()
        self.assertFalse(v.initialized)
        self.assertFalse(v.child.initialized)

    def test_order_is_limited(self):
        self.assertEqual(VMedian(order=-3).order, 0)
        self.assertEqual(VMedian(order=50).order, 10)

    def test_the_shape_follows_the_frames(self):
        v = VMedian()
        v.add(np.zeros((3, 7), dtype=np.uint8))
        self.assertEqual(v.shape, (3, 7))
        v.add(np.zeros((5, 2), dtype=np.uint8))       # a different size
        self.assertEqual(v.shape, (5, 2))
        self.assertFalse(v.initialized)

    def test_the_shape_can_be_given(self):
        self.assertEqual(VMedian(shape=(8, 9)).shape, (8, 9))


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
