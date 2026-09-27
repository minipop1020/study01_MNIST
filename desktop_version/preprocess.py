"""MNIST 방식 전처리 (tkinter 의존 없음)

같은 알고리즘이 세 곳에 있다. 하나를 바꾸면 나머지도 같이 바꾼다.
  - desktop_version/preprocess.py  (이 파일)
  - desktop_version/index.html     toMnist
  - web_version/preprocess.js      toMnist

1. 값 > 20 인 픽셀의 경계 상자를 구한다.
2. 긴 변이 20px 이 되도록 비율을 유지해 면적 평균으로 축소한다.
3. 무게중심이 28×28 의 (14,14) 에 오도록 놓는다.
반올림은 항상 floor(v + 0.5) 이다 (파이썬 round 는 짝수 반올림이라 쓰지 않는다).
"""
import numpy as np


def round_half_up(v):
    return np.floor(np.asarray(v, dtype=np.float64) + 0.5)


def _axis_weights(n_src: int, n_out: int) -> np.ndarray:
    """출력 칸 o 가 소스 구간 [o*n_src/n_out, (o+1)*n_src/n_out) 과 겹치는 길이 (n_out, n_src)."""
    step = n_src / n_out
    lo = np.arange(n_out)[:, None] * step
    hi = lo + step
    i = np.arange(n_src)[None, :]
    return np.clip(np.minimum(i + 1, hi) - np.maximum(i, lo), 0, None)


def area_downscale(src: np.ndarray, sw: int, sh: int) -> np.ndarray:
    """(bh, bw) 이미지를 (sh, sw) 로 정확한 면적 평균 축소. 가로 먼저, 세로 다음."""
    bh, bw = src.shape
    wx = _axis_weights(bw, sw)
    wy = _axis_weights(bh, sh)
    rows = src.astype(np.float64) @ wx.T          # (bh, sw)
    out = (wy @ rows) / ((bw / sw) * (bh / sh))   # (sh, sw)
    return np.clip(round_half_up(out), 0, 255).astype(np.uint8)


def to_mnist(arr: np.ndarray) -> np.ndarray | None:
    """(H, W) 흑백 비트맵 → 28×28 uint8. 그린 것이 없으면 None."""
    ys, xs = np.nonzero(arr > 20)
    if len(xs) == 0:
        return None
    x0, y0 = xs.min(), ys.min()
    bw, bh = xs.max() - x0 + 1, ys.max() - y0 + 1

    scale = 20 / max(bw, bh)
    sw = max(1, int(round_half_up(bw * scale)))
    sh = max(1, int(round_half_up(bh * scale)))
    small = area_downscale(arr[y0:y0 + bh, x0:x0 + bw], sw, sh)

    # 정수화된 축소 결과로 무게중심을 구해 (14,14) 로 옮긴다.
    s = small.astype(np.float64)
    total = s.sum()
    if total == 0:                                # 축소하며 모두 0 이 된 경우
        return None
    cx = (np.arange(sw)[None, :] * s).sum() / total
    cy = (np.arange(sh)[:, None] * s).sum() / total
    ox = int(np.floor(14 - cx))
    oy = int(np.floor(14 - cy))

    out = np.zeros((28, 28), dtype=np.uint8)
    for y in range(sh):
        ty = y + oy
        if not 0 <= ty < 28:
            continue
        for x in range(sw):
            tx = x + ox
            if 0 <= tx < 28:
                out[ty, tx] = small[y, x]
    return out


def upscale10(img28: np.ndarray) -> np.ndarray:
    """28×28 → 280×280 최근접 이웃 확대. big[y][x] = img[y//10][x//10]"""
    return np.repeat(np.repeat(np.asarray(img28, dtype=np.uint8), 10, axis=0), 10, axis=1)
