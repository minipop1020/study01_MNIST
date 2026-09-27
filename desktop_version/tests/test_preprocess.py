"""preprocess.py 검사: python -m unittest tests.test_preprocess -v (desktop_version 폴더에서)"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from preprocess import area_downscale, round_half_up, to_mnist, upscale10  # noqa: E402


def center_of_mass(img: np.ndarray) -> tuple[float, float]:
    a = img.astype(np.float64)
    total = a.sum()
    cy = (np.arange(a.shape[0])[:, None] * a).sum() / total
    cx = (np.arange(a.shape[1])[None, :] * a).sum() / total
    return cx, cy


class RoundTest(unittest.TestCase):
    def test_round_half_up(self):
        # 파이썬 round 는 0.5, 2.5 를 짝수로 보낸다. 여기서는 항상 올린다.
        np.testing.assert_array_equal(
            round_half_up(np.array([0.5, 1.5, 2.5, -0.5])), [1, 2, 3, 0])


class AreaDownscaleTest(unittest.TestCase):
    def test_area_downscale_exact(self):
        src = np.array([[0, 90, 180],
                        [90, 180, 255],
                        [180, 255, 255]], dtype=np.uint8)
        # 출력 한 칸이 소스 1.5×1.5 를 덮는다. 손으로 계산한 면적 평균:
        # (0,0): (0 + 90*.5 + 90*.5 + 180*.25) / 2.25 = 60
        # (0,1): (90*.5 + 180 + (180*.5 + 255)*.5) / 2.25 = 176.67 → 177
        # (1,0): 대칭이라 177
        # (1,1): ((180*.5 + 255)*.5 + 255*.5 + 255) / 2.25 = 246.67 → 247
        np.testing.assert_array_equal(area_downscale(src, 2, 2),
                                      [[60, 177], [177, 247]])

    def test_area_downscale_constant(self):
        src = np.full((5, 7), 200, dtype=np.uint8)
        out = area_downscale(src, 3, 2)
        self.assertEqual(out.shape, (2, 3))
        self.assertEqual(out.dtype, np.uint8)
        self.assertTrue((out == 200).all())


class ToMnistTest(unittest.TestCase):
    def rect(self) -> np.ndarray:
        img = np.zeros((280, 280), dtype=np.uint8)
        img[150:260, 40:120] = 255
        return img

    def test_empty_returns_none(self):
        self.assertIsNone(to_mnist(np.zeros((280, 280), np.uint8)))

    def test_center_of_mass(self):
        out = to_mnist(self.rect())
        self.assertEqual(out.shape, (28, 28))
        self.assertEqual(out.dtype, np.uint8)
        self.assertGreater(int(out.sum()), 0)
        cx, cy = center_of_mass(out)
        self.assertLess(abs(cx - 14), 1)
        self.assertLess(abs(cy - 14), 1)

    def test_longer_side_is_20(self):
        out = to_mnist(self.rect())
        ys, xs = np.nonzero(out)
        self.assertEqual(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1), 20)

    def test_single_dot(self):
        img = np.zeros((280, 280), np.uint8)
        img[100, 100] = 255
        out = to_mnist(img)
        self.assertIsNotNone(out)
        self.assertGreater(int(out.sum()), 0)

    def test_thin_line(self):
        img = np.zeros((280, 280), np.uint8)
        img[20:251, 50] = 255
        out = to_mnist(img)
        self.assertIsNotNone(out)
        ys, xs = np.nonzero(out)
        self.assertGreaterEqual(xs.max() - xs.min() + 1, 1)
        self.assertEqual(ys.max() - ys.min() + 1, 20)


class Upscale10Test(unittest.TestCase):
    def test_upscale10(self):
        img = np.random.default_rng(0).integers(0, 256, (28, 28), dtype=np.uint8)
        big = upscale10(img)
        self.assertEqual(big.shape, (280, 280))
        for y, x in [(0, 0), (9, 9), (10, 0), (137, 251), (279, 279)]:
            self.assertEqual(big[y, x], img[y // 10, x // 10])


if __name__ == "__main__":
    unittest.main()
