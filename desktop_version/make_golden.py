"""검증 데이터 → ../web_version/golden.js

공식 시험 이미지 200장에 대해 파이썬 전처리 결과와 모델 확률을 저장한다.
web_version/tests.html 이 JS 구현이 같은 결과를 내는지 이것으로 확인한다.
train.py 로 모델을 다시 만들었으면 이것도 다시 실행한다.
"""
import base64
import json
from pathlib import Path

import numpy as np

from mnist_data import load_mnist
from mnist_model import MLP
from preprocess import to_mnist, upscale10

HERE = Path(__file__).resolve().parent


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=np.uint8).tobytes()).decode("ascii")


def main():
    _, _, X_te, y_te = load_mnist(HERE / "data")
    model = MLP.load(HERE / "model.npz")
    picks = np.random.default_rng(0).choice(len(X_te), 200, replace=False)

    items = []
    for i in picks:
        src = X_te[i].reshape(28, 28)
        pre = to_mnist(upscale10(src))            # 앱 입력과 같은 조건: 280×280 으로 키운 뒤 전처리
        probs = model.predict_proba(pre.reshape(784) / 255)
        items.append({"label": int(y_te[i]), "src": b64(src), "pre": b64(pre),
                      "probs": [round(float(p), 7) for p in probs]})

    out = HERE.parent / "web_version" / "golden.js"
    out.write_text("window.MNIST_GOLDEN = " + json.dumps({"items": items}) + ";\n", encoding="utf-8")
    acc = np.mean([np.argmax(it["probs"]) == it["label"] for it in items])
    print(f"저장: {out}  ({len(items)}장, 파이썬 정확도 {acc * 100:.1f}%)")


if __name__ == "__main__":
    main()
