"""mnist_data.py 검사 (네트워크 다운로드는 검사하지 않는다)"""
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mnist_data import load_my_csv, parse_idx  # noqa: E402


class ParseIdxTest(unittest.TestCase):
    def test_parse_images(self):
        body = bytes(i % 256 for i in range(2 * 784))
        arr = parse_idx(struct.pack(">IIII", 0x803, 2, 28, 28) + body)
        self.assertEqual(arr.shape, (2, 784))
        self.assertEqual(arr.dtype, np.uint8)
        np.testing.assert_array_equal(arr.ravel(), np.frombuffer(body, np.uint8))

    def test_parse_labels(self):
        arr = parse_idx(struct.pack(">II", 0x801, 3) + bytes([7, 0, 9]))
        np.testing.assert_array_equal(arr, [7, 0, 9])

    def test_bad_magic(self):
        with self.assertRaises(ValueError):
            parse_idx(struct.pack(">II", 0x1234, 0))


class MyCsvTest(unittest.TestCase):
    def test_my_csv_missing(self):
        X, y = load_my_csv(Path(tempfile.gettempdir()) / "no_such_mnist_file.csv")
        self.assertEqual(X.shape, (0, 784))
        self.assertEqual(y.shape, (0,))

    def test_my_csv_rows(self):
        header = ",".join(["label"] + [f"pixel{i}" for i in range(784)])
        row1 = ",".join(["3"] + ["0"] * 783 + ["255"])
        row2 = ",".join(["8"] + ["7"] * 784)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "my.csv"
            p.write_text("\n".join([header, row1, row2]) + "\n", encoding="utf-8")
            X, y = load_my_csv(p)
        self.assertEqual(X.shape, (2, 784))
        self.assertEqual(X.dtype, np.uint8)
        np.testing.assert_array_equal(y, [3, 8])
        self.assertEqual(X[0, 783], 255)
        self.assertEqual(X[0, 0], 0)
        self.assertTrue((X[1] == 7).all())


if __name__ == "__main__":
    unittest.main()
