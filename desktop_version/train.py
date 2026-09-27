"""MNIST 학습 → model.npz + ../web_version/model.js

실행: train.bat 더블클릭
공식 학습 데이터 60,000장에 my_mnist.csv(직접 모은 글씨)를 더해 학습한다.
배치마다 augment.py 로 글씨를 무작위 변형해 넣는다(마우스 글씨 대비).
"""
import time
from pathlib import Path

import numpy as np

from augment import augment
from mnist_data import load_mnist, load_my_csv
from mnist_model import MLP

HERE = Path(__file__).resolve().parent
EPOCHS, BATCH, VAL_SIZE, SEED = 30, 64, 5000, 0
SIZES = (784, 256, 128, 10)
LR_DECAY = 0.93          # 에폭마다 학습률에 곱한다


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
    X_fit = X[tr]
    y_fit = y[tr]

    model = MLP(SIZES, seed=SEED)
    start = time.time()
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(len(X_fit))
        losses = [model.train_step(augment(X_fit[b], rng, y_fit[b]) / 255, y_fit[b])
                  for b in (order[i:i + BATCH] for i in range(0, len(order), BATCH))]
        print(f"epoch {epoch}/{EPOCHS}  loss {np.mean(losses):.3f}  "
              f"val {accuracy(model, X_val, y_val) * 100:.1f}%")
        model.lr *= LR_DECAY

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
