"""app.py 의 모델 불러오기 검사 (창은 띄우지 않는다)"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import app  # noqa: E402
from mnist_model import MLP  # noqa: E402


class LoadModelTest(unittest.TestCase):
    def load_from(self, path: Path):
        old = app.MODEL_PATH
        app.MODEL_PATH = path
        try:
            return app.App.load_model(None)      # self 는 쓰지 않는다
        finally:
            app.MODEL_PATH = old

    def test_corrupt_model_files_return_none(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            good = d / "good.npz"
            MLP(seed=0).save(good)
            cases = {
                "empty": b"",
                "truncated": good.read_bytes()[:500],
                "not_zip": b"hello",
            }
            for name, data in cases.items():
                p = d / f"{name}.npz"
                p.write_bytes(data)
                self.assertIsNone(self.load_from(p), name)
            np.savez(d / "no_weights.npz", x=np.zeros(3))
            self.assertIsNone(self.load_from(d / "no_weights.npz"), "no_weights")
            np.savez(d / "bad_shape.npz", W0=np.zeros((100, 10), np.float32), b0=np.zeros(10, np.float32))
            self.assertIsNone(self.load_from(d / "bad_shape.npz"), "bad_shape")
            self.assertIsNotNone(self.load_from(good), "good")

    def test_missing_model_returns_none(self):
        self.assertIsNone(self.load_from(Path(tempfile.gettempdir()) / "no_such_model.npz"))


if __name__ == "__main__":
    unittest.main()
