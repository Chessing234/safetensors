import unittest

import numpy as np

from safetensors.numpy import load, save


class NumpyContiguousTest(unittest.TestCase):
    def test_fortran_array_roundtrips(self):
        values = np.arange(6, dtype=np.float32).reshape(2, 3).T
        self.assertFalse(values.flags.c_contiguous)

        loaded = load(save({"w": values}))["w"]

        np.testing.assert_array_equal(loaded, values)
        self.assertEqual(loaded.shape, (3, 2))

    def test_strided_slice_roundtrips(self):
        base = np.arange(12, dtype=np.int64).reshape(3, 4)
        view = base[:, ::2]
        self.assertFalse(view.flags.c_contiguous)

        loaded = load(save({"w": view}))["w"]

        np.testing.assert_array_equal(loaded, view)
        self.assertEqual(tuple(loaded.shape), (3, 2))


if __name__ == "__main__":
    unittest.main()
