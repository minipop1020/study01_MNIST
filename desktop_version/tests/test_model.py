"""mnist_model.py 검사"""
import base64
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mnist_model import MLP  # noqa: E402


class GradientTest(unittest.TestCase):
    def test_gradient_check(self):
        m = MLP((5, 4, 3, 2), seed=1)
        # 수치 미분은 float64 로 해야 정확하다.
        m.Ws = [w.astype(np.float64) for w in m.Ws]
        m.bs = [b.astype(np.float64) for b in m.bs]
        rng = np.random.default_rng(2)
        x = rng.normal(size=(3, 5))
        y = np.array([0, 1, 1])
        _, grads = m.loss_and_grads(x, y)

        eps = 1e-5
        for layer, (dW, db) in enumerate(grads):
            for param, g in ((m.Ws[layer], dW), (m.bs[layer], db)):
                num = np.zeros_like(param)
                for idx in np.ndindex(param.shape):
                    old = param[idx]
                    param[idx] = old + eps
                    lp, _ = m.loss_and_grads(x, y)
                    param[idx] = old - eps
                    lm, _ = m.loss_and_grads(x, y)
                    param[idx] = old
                    num[idx] = (lp - lm) / (2 * eps)
                rel = np.abs(num - g).max() / max(1e-8, np.abs(num).max() + np.abs(g).max())
                self.assertLess(rel, 1e-4, f"layer {layer}")


class TrainTest(unittest.TestCase):
    def test_loss_decreases(self):
        rng = np.random.default_rng(3)
        y = rng.integers(0, 2, 200)
        x = rng.random((200, 784)) * 0.3
        x[y == 1, :100] += 0.5                      # 클래스 1 은 앞쪽 픽셀이 밝다
        m = MLP(seed=0)
        first = m.train_step(x[:64], y[:64])
        for i in range(50):
            s = (i * 64) % 136
            last = m.train_step(x[s:s + 64], y[s:s + 64])
        self.assertLess(last, first * 0.5)


class IoTest(unittest.TestCase):
    def test_save_load_roundtrip(self):
        m = MLP(seed=4)
        x = np.random.default_rng(5).random((3, 784))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.npz"
            m.save(p)
            m2 = MLP.load(p)
        np.testing.assert_array_equal(m.predict_proba(x), m2.predict_proba(x))

    def test_export_js_base64(self):
        m = MLP(seed=6)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "model.js"
            m.export_js(p, {"testAccuracy": 0.978})
            text = p.read_text(encoding="utf-8")
        prefix = "window.MNIST_MODEL = "
        self.assertTrue(text.startswith(prefix))
        spec = json.loads(text[len(prefix):].strip().rstrip(";"))
        self.assertEqual(spec["version"], 1)
        self.assertEqual(spec["testAccuracy"], 0.978)
        self.assertEqual(len(spec["layers"]), 3)
        for i, layer in enumerate(spec["layers"]):
            W = np.frombuffer(base64.b64decode(layer["W"]), "<f4").reshape(layer["inSize"], layer["outSize"])
            b = np.frombuffer(base64.b64decode(layer["b"]), "<f4")
            np.testing.assert_array_equal(W, m.Ws[i])
            np.testing.assert_array_equal(b, m.bs[i])
        self.assertEqual([l["inSize"] for l in spec["layers"]], [784, 128, 64])
        self.assertEqual([l["outSize"] for l in spec["layers"]], [128, 64, 10])

    def test_predict_single_sums_to_one(self):
        p = MLP(seed=7).predict_proba(np.zeros(784))
        self.assertEqual(p.shape, (10,))
        self.assertAlmostEqual(float(p.sum()), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
