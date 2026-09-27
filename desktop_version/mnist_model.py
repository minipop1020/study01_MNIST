"""numpy 로 만든 다층 퍼셉트론 (784 → 128 → 64 → 10)

은닉층 ReLU, 출력층 softmax, 손실은 교차 엔트로피, 최적화는 Adam.
가중치 W 는 (in, out) 모양이라 y = x @ W + b 이다. web_version/mlp.js 도 같은 규칙을 쓴다.
"""
import base64
import json
from pathlib import Path

import numpy as np


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class MLP:
    def __init__(self, sizes=(784, 128, 64, 10), seed=0):
        rng = np.random.default_rng(seed)
        self.sizes = tuple(sizes)
        # He 초기화
        self.Ws = [(rng.normal(size=(i, o)) * np.sqrt(2 / i)).astype(np.float32)
                   for i, o in zip(sizes[:-1], sizes[1:])]
        self.bs = [np.zeros(o, dtype=np.float32) for o in sizes[1:]]
        self.lr, self.beta1, self.beta2, self.eps = 1e-3, 0.9, 0.999, 1e-8
        self._m = self._v = None
        self._t = 0

    # ---------- 순전파 / 역전파 ----------
    def _forward(self, x):
        """각 층의 활성값 목록을 돌려준다. acts[0] = x, acts[-1] = 확률."""
        acts = [x]
        for i, (W, b) in enumerate(zip(self.Ws, self.bs)):
            z = acts[-1] @ W + b
            acts.append(softmax(z) if i == len(self.Ws) - 1 else np.maximum(z, 0))
        return acts

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self._forward(np.atleast_2d(x))[-1]

    def loss_and_grads(self, x, y):
        acts = self._forward(x)
        probs = acts[-1]
        n = len(y)
        loss = float(-np.log(probs[np.arange(n), y] + 1e-12).mean())

        delta = probs.copy()
        delta[np.arange(n), y] -= 1
        delta /= n
        grads = []
        for i in range(len(self.Ws) - 1, -1, -1):
            grads.append((acts[i].T @ delta, delta.sum(axis=0)))
            if i > 0:
                delta = (delta @ self.Ws[i].T) * (acts[i] > 0)
        return loss, grads[::-1]

    def train_step(self, x, y) -> float:
        loss, grads = self.loss_and_grads(x.astype(np.float32), y)
        params = [p for pair in zip(self.Ws, self.bs) for p in pair]
        flat = [g for pair in grads for g in pair]
        if self._m is None:
            self._m = [np.zeros_like(p) for p in params]
            self._v = [np.zeros_like(p) for p in params]
        self._t += 1
        c1 = 1 - self.beta1 ** self._t
        c2 = 1 - self.beta2 ** self._t
        for p, g, m, v in zip(params, flat, self._m, self._v):
            m *= self.beta1
            m += (1 - self.beta1) * g
            v *= self.beta2
            v += (1 - self.beta2) * g * g
            p -= (self.lr * (m / c1) / (np.sqrt(v / c2) + self.eps)).astype(p.dtype)
        return loss

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """x 가 (784,) 이면 (10,), (n, 784) 이면 (n, 10)."""
        x = np.asarray(x, dtype=np.float32)
        probs = self.forward(x)
        return probs[0] if x.ndim == 1 else probs

    # ---------- 저장 ----------
    def save(self, path):
        arrays = {}
        for i, (W, b) in enumerate(zip(self.Ws, self.bs)):
            arrays[f"W{i}"] = W
            arrays[f"b{i}"] = b
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path) -> "MLP":
        with np.load(path) as f:
            n = len([k for k in f.files if k.startswith("W")])
            Ws = [f[f"W{i}"].astype(np.float32) for i in range(n)]
            bs = [f[f"b{i}"].astype(np.float32) for i in range(n)]
        m = cls(sizes=(Ws[0].shape[0], *[W.shape[1] for W in Ws]))
        m.Ws, m.bs = Ws, bs
        return m

    def export_js(self, path, meta: dict):
        """window.MNIST_MODEL = {...}; 형식. 가중치는 float32 little-endian base64."""
        def b64(a):
            return base64.b64encode(np.ascontiguousarray(a, dtype="<f4").tobytes()).decode("ascii")

        spec = {"version": 1, **meta, "layers": [
            {"inSize": int(W.shape[0]), "outSize": int(W.shape[1]), "W": b64(W), "b": b64(b)}
            for W, b in zip(self.Ws, self.bs)]}
        Path(path).write_text("window.MNIST_MODEL = " + json.dumps(spec) + ";\n", encoding="utf-8")
