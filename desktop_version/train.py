"""MNIST 학습 → model.npz + ../web_version/model.js

실행: train.bat 더블클릭
공식 학습 데이터 60,000장에 my_mnist.csv(직접 모은 글씨)를 더해 학습한다.
"""
import time
from pathlib import Path

import numpy as np

from mnist_data import load_mnist, load_my_csv
from mnist_model import MLP

HERE = Path(__file__).resolve().parent
EPOCHS, BATCH, VAL_SIZE, SEED = 10, 64, 5000, 0


def accuracy(model: MLP, X: np.ndarray, y: np.ndarray) -> float:
    pred = np.concatenate([model.predict_proba(X[i:i + 2000] / 255).argmax(axis=1)
                           for i in range(0, len(X), 2000)])
    return float((pred == y).mean())


def main():
    X_tr, y_tr, X_te, y_te = load_mnist(HERE / "data")
    X_my, y_my = load_my_csv(HERE / "my_mnist.csv")
    print(f"공식 학습 데이터 {len(X_tr)}장 + 직접 모은 글씨 {len(X_my)}장")
    X = np.concatenate([X_tr, X_my])
    y = np.concatenate([y_tr.astype(np.int64), y_my])

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(X))
    val, tr = idx[:VAL_SIZE], idx[VAL_SIZE:]
    X_val, y_val = X[val], y[val]
    X_fit = X[tr].astype(np.float32) / 255
    y_fit = y[tr]

    model = MLP(seed=SEED)
    start = time.time()
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(len(X_fit))
        losses = [model.train_step(X_fit[b], y_fit[b])
                  for b in (order[i:i + BATCH] for i in range(0, len(order), BATCH))]
        print(f"epoch {epoch}/{EPOCHS}  loss {np.mean(losses):.3f}  "
              f"val {accuracy(model, X_val, y_val) * 100:.1f}%")

    test_acc = accuracy(model, X_te, y_te)
    print(f"test {test_acc * 100:.1f}%  ({time.time() - start:.0f}초)")
    if test_acc < 0.97:
        print("경고: test 정확도가 97% 미만입니다.")

    model.save(HERE / "model.npz")
    js = HERE.parent / "web_version" / "model.js"
    js.parent.mkdir(exist_ok=True)
    model.export_js(js, {"testAccuracy": round(test_acc, 4)})
    print(f"저장: {HERE / 'model.npz'}\n저장: {js}")


if __name__ == "__main__":
    main()
