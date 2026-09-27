"""공식 MNIST 데이터 다운로드·파싱과 직접 모은 my_mnist.csv 읽기"""
import csv
import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np

MIRRORS = [
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
]
FILES = {
    "train_x": "train-images-idx3-ubyte.gz",
    "train_y": "train-labels-idx1-ubyte.gz",
    "test_x": "t10k-images-idx3-ubyte.gz",
    "test_y": "t10k-labels-idx1-ubyte.gz",
}


def parse_idx(data: bytes) -> np.ndarray:
    """IDX 형식 바이트 → 이미지 (n, 784) 또는 라벨 (n,)."""
    magic = struct.unpack(">I", data[:4])[0]
    if magic == 0x803:
        n, rows, cols = struct.unpack(">III", data[4:16])
        return np.frombuffer(data, np.uint8, n * rows * cols, 16).reshape(n, rows * cols)
    if magic == 0x801:
        n = struct.unpack(">I", data[4:8])[0]
        return np.frombuffer(data, np.uint8, n, 8).copy()
    raise ValueError(f"IDX 매직 넘버가 올바르지 않습니다: {magic:#x}")


def _download(name: str, dest: Path):
    for base in MIRRORS:
        try:
            print(f"다운로드: {base}{name}")
            with urllib.request.urlopen(base + name, timeout=60) as r:
                dest.write_bytes(r.read())
            return
        except OSError as ex:
            print(f"  실패: {ex}")
    raise SystemExit(
        f"{name} 을(를) 받지 못했습니다.\n"
        f"브라우저로 위 주소에서 직접 받아 {dest.parent} 폴더에 넣은 뒤 다시 실행하세요.")


def load_mnist(cache_dir: Path):
    """(X_train, y_train, X_test, y_test). 파일이 없으면 cache_dir 에 내려받는다."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for key, name in FILES.items():
        path = cache_dir / name
        if not path.exists():
            _download(name, path)
        out[key] = parse_idx(gzip.decompress(path.read_bytes()))
    return out["train_x"], out["train_y"], out["test_x"], out["test_y"]


def load_my_csv(path: Path):
    """label,pixel0..pixel783 CSV → (X uint8 (n,784), y int64 (n,)). 파일이 없으면 빈 배열."""
    path = Path(path)
    if not path.exists():
        return np.zeros((0, 784), np.uint8), np.zeros(0, np.int64)
    with path.open(newline="") as f:
        reader = csv.reader(f)
        next(reader, None)                       # 헤더 건너뛰기
        rows = [list(map(int, r)) for r in reader if r]
    if not rows:
        return np.zeros((0, 784), np.uint8), np.zeros(0, np.int64)
    arr = np.array(rows, dtype=np.int64)
    return arr[:, 1:].astype(np.uint8), arr[:, 0]
