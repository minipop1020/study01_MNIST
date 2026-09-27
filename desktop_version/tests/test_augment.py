"""augment.py 검사"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from augment import add_flag, affine, augment  # noqa: E402


class AffineTest(unittest.TestCase):
    def setUp(self):
        self.img = np.random.default_rng(0).uniform(0, 255, (1, 28, 28)).astype(np.float32)

    def test_identity_keeps_image(self):
        out = affine(self.img, np.eye(2)[None], np.zeros((1, 2)))
        np.testing.assert_allclose(out, self.img, atol=1e-3)

    def test_integer_shift_moves_pixels(self):
        out = affine(self.img, np.eye(2)[None], np.array([[2.0, 1.0]]))   # 오른쪽 2, 아래 1
        np.testing.assert_allclose(out[0, 1:, 2:], self.img[0, :-1, :-2], atol=1e-3)
        self.assertTrue(np.all(out[0, 0, :] == 0) and np.all(out[0, :, :2] == 0))


class AugmentTest(unittest.TestCase):
    def test_shape_range_and_randomness(self):
        X = np.zeros((5, 784), np.uint8)
        X[:, 28 * 10 + 8:28 * 10 + 20] = 255                     # 가로 획
        rng = np.random.default_rng(1)
        a, b = augment(X, rng), augment(X, rng)
        self.assertEqual(a.shape, (5, 784))
        self.assertEqual(a.dtype, np.float32)
        self.assertTrue(0 <= a.min() and a.max() <= 255)
        self.assertGreater(a.sum(axis=1).min(), 0)              # 획이 사라지지 않는다
        self.assertFalse(np.allclose(a, b))

    def test_flag_added_only_to_ones(self):
        img = np.zeros((28, 28), np.float32)
        img[6:24, 14] = 255                                       # 세로 막대
        flagged = img.copy()
        add_flag(flagged, np.random.default_rng(0))
        self.assertGreater(flagged[6:12, 8:14].sum(), 0)          # 왼쪽 위에 깃발이 생긴다
        np.testing.assert_array_equal(flagged[6:24, 14], img[6:24, 14])  # 막대는 그대로

        X = np.stack([img.ravel()] * 2).astype(np.uint8)
        rng = np.random.default_rng(3)
        out = augment(X, rng, y=np.array([7, 1]), frac=0.0, flag_prob=1.0)
        np.testing.assert_array_equal(out[0], X[0])               # 7 은 변형도 깃발도 없다
        self.assertGreater(out[1].reshape(28, 28)[6:12, 8:14].sum(), 0)


if __name__ == "__main__":
    unittest.main()
