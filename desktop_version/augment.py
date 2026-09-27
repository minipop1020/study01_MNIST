"""학습용 데이터 증강: 28×28 글씨를 무작위로 기울이고, 늘이고, 옮기고, 획 굵기를 바꾼다.

마우스로 그린 글씨는 MNIST 보다 모양이 제각각이라(동그라미가 큰 6 등) 원본만으로 학습하면
처음 보는 비율의 글씨를 틀린다. 매 배치마다 새로 변형해 넣어 이런 차이에 익숙해지게 한다.
"""
import numpy as np

CENTER = 13.5


def affine(imgs: np.ndarray, mats: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """imgs (n,28,28) float → 출력 좌표 p 에 입력 좌표 mats⁻¹(p - c - shift) + c 의 값을 쌍선형 보간으로 넣는다."""
    n = len(imgs)
    inv = np.linalg.inv(mats)                                    # (n,2,2)
    ys, xs = np.mgrid[0:28, 0:28].astype(np.float32)
    px = xs[None] - CENTER - shifts[:, 0, None, None]            # (n,28,28)
    py = ys[None] - CENTER - shifts[:, 1, None, None]
    sx = inv[:, 0, 0, None, None] * px + inv[:, 0, 1, None, None] * py + CENTER
    sy = inv[:, 1, 0, None, None] * px + inv[:, 1, 1, None, None] * py + CENTER

    padded = np.pad(imgs, ((0, 0), (1, 1), (1, 1))).reshape(n, -1)   # 바깥은 0
    x0 = np.floor(sx)
    y0 = np.floor(sy)
    fx, fy = sx - x0, sy - y0

    def at(yy, xx):
        yy = np.clip(yy + 1, 0, 29).astype(np.int64)
        xx = np.clip(xx + 1, 0, 29).astype(np.int64)
        return np.take_along_axis(padded, (yy * 30 + xx).reshape(n, -1), axis=1).reshape(n, 28, 28)

    return ((1 - fy) * ((1 - fx) * at(y0, x0) + fx * at(y0, x0 + 1))
            + fy * ((1 - fx) * at(y0 + 1, x0) + fx * at(y0 + 1, x0 + 1)))


def _neighbors(imgs: np.ndarray):
    p = np.pad(imgs, ((0, 0), (1, 1), (1, 1)))
    return p[:, :-2, 1:-1], p[:, 2:, 1:-1], p[:, 1:-1, :-2], p[:, 1:-1, 2:]


def add_flag(img: np.ndarray, rng: np.random.Generator) -> None:
    """1 의 맨 위에서 왼쪽 아래로 짧은 깃발 획을 긋는다(제자리 수정).

    MNIST 의 1 은 대부분 막대 하나라, 깃발 달린 1 을 7 로 읽었다.
    """
    rows = np.nonzero((img > 128).any(axis=1))[0]
    if len(rows) == 0:
        return
    top = rows[0]
    x0 = float(np.nonzero(img[top] > 128)[0].mean())
    dx = rng.uniform(2.5, 5.5)
    dy = dx * rng.uniform(0.4, 0.9)
    ys, xs = np.mgrid[0:28, 0:28]
    t = np.linspace(0, 1, 24)[:, None, None]
    px, py = x0 - dx * t, top + dy * t                      # 선분 위의 점들
    dist = np.sqrt((xs - px) ** 2 + (ys - py) ** 2).min(axis=0)
    np.maximum(img, np.clip(1.6 - dist, 0, 1) * 255, out=img)


def augment(X: np.ndarray, rng: np.random.Generator, y=None,
            max_angle=15, max_shear=0.3, frac=0.5, flag_prob=0.35) -> np.ndarray:
    """X (n,784) 0–255 → 변형한 (n,784) float32 0–255. frac 비율만 변형하고 나머지는 원본 그대로 둔다.

    y(라벨)를 주면 1 의 flag_prob 비율에 깃발 획을 붙인다.
    """
    n = len(X)
    imgs = X.reshape(n, 28, 28).astype(np.float32)
    if y is not None:
        for i in np.nonzero((np.asarray(y) == 1) & (rng.random(n) < flag_prob))[0]:
            add_flag(imgs[i], rng)

    angle = np.radians(rng.uniform(-max_angle, max_angle, n))
    shear = rng.uniform(-max_shear, max_shear, n)
    scale_x = rng.uniform(0.75, 1.15, n)      # 가로세로를 따로 늘여 비율이 다른 글씨를 만든다
    scale_y = rng.uniform(0.8, 1.15, n)
    c, s = np.cos(angle), np.sin(angle)
    rot = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], 1)
    shr = np.zeros((n, 2, 2))
    shr[:, 0, 0] = shr[:, 1, 1] = 1
    shr[:, 0, 1] = shear
    scl = np.zeros((n, 2, 2))
    scl[:, 0, 0], scl[:, 1, 1] = scale_x, scale_y
    mats = rot @ shr @ scl
    shifts = rng.uniform(-2, 2, (n, 2))
    out = affine(imgs, mats, shifts)

    # 획 굵기: 일부는 두껍게(이웃 최댓값), 일부는 가늘게(이웃 최솟값) — 원래 값과 반씩 섞는다.
    kind = rng.random(n)
    thick, thin = kind < 0.2, kind > 0.85
    nb = _neighbors(out)
    if thick.any():
        grown = np.maximum.reduce([out, *nb])
        out[thick] = 0.5 * (out[thick] + grown[thick])
    if thin.any():
        shrunk = np.minimum.reduce([out, *nb])
        out[thin] = 0.5 * (out[thin] + shrunk[thin])

    out = np.clip(out, 0, 255)
    keep = rng.random(n) >= frac
    out[keep] = imgs[keep]
    return out.reshape(n, 784).astype(np.float32)
