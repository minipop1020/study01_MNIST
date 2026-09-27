"""손글씨 숫자 입력기 + 인식기 (MNIST 형식)

숫자를 그리고 0~9 라벨을 누르면 28x28 흑백 이미지로 변환해
같은 폴더의 my_mnist.csv 에 한 줄씩 저장합니다.
CSV 형식: label, pixel0 ... pixel783  (Kaggle mnist_train.csv 와 동일)
그림을 그리면 model.npz(train.bat 으로 만든 모델)로 바로 인식해 오른쪽에 보여준다.

실행: run.bat 더블클릭  (.py 파일은 파이썬에 연결돼 있지 않다)
필요 라이브러리: numpy, pillow  (pip install numpy pillow)
"""
import csv
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from mnist_model import MLP
from preprocess import to_mnist

SIZE = 280                       # 그리기 캔버스 크기 (px)
PREVIEW = 140                    # 미리보기 크기 (28x28 을 5배 확대)
CSV_PATH = Path(__file__).parent / "my_mnist.csv"
MODEL_PATH = Path(__file__).parent / "model.npz"
TOP_K = 3                        # 보여 줄 후보 수
HINT = "그림을 그리면 자동으로 인식합니다."
BAR_W, ROW_H = 150, 24           # 후보 막대 크기


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("손글씨 숫자 입력기 · 인식기 (MNIST)")
        root.resizable(False, False)

        self.strokes: list[dict] = []    # 되돌리기용 획 목록
        self.current: dict | None = None
        self.counts = self.load_counts()
        self.pixels = None                # 마지막으로 계산한 28x28
        self.model = self.load_model()

        # ---- 왼쪽: 그리기 영역 ----
        left = tk.Frame(root, padx=12, pady=12)
        left.grid(row=0, column=0, sticky="n")
        tk.Label(left, text="여기에 그리세요").pack(anchor="w")
        self.canvas = tk.Canvas(left, width=SIZE, height=SIZE, bg="black",
                                highlightthickness=0, cursor="crosshair")
        self.canvas.pack()
        self.canvas.bind("<ButtonPress-1>", self.on_down)
        self.canvas.bind("<B1-Motion>", self.on_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_up)

        brush_row = tk.Frame(left)
        brush_row.pack(fill="x", pady=(8, 0))
        tk.Label(brush_row, text="붓 굵기").pack(side="left")
        self.brush = tk.Scale(brush_row, from_=10, to=36, orient="horizontal",
                              showvalue=True, length=200)
        self.brush.set(22)
        self.brush.pack(side="left", padx=6)

        btn_row = tk.Frame(left)
        btn_row.pack(fill="x", pady=(8, 0))
        tk.Button(btn_row, text="지우기 (Esc)", command=self.clear).pack(side="left")
        tk.Button(btn_row, text="되돌리기 (Ctrl+Z)", command=self.undo).pack(side="left", padx=6)
        tk.Button(btn_row, text="인식하기", command=lambda: self.recognize(True)).pack(side="left")

        # ---- 오른쪽: 미리보기 + 라벨 ----
        right = tk.Frame(root, padx=12, pady=12)
        right.grid(row=0, column=1, sticky="n")
        tk.Label(right, text="28×28 전처리 미리보기").pack(anchor="w")
        self.preview = tk.Label(right, bg="black", width=PREVIEW, height=PREVIEW)
        self.preview.pack(anchor="w")

        # ---- 인식 결과 ----
        tk.Label(right, text="인식 결과", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(14, 0))
        self.digit = tk.Label(right, text="?", font=("Segoe UI", 54))
        self.digit.pack(anchor="w")
        self.status = tk.Label(right, text=HINT, fg="#6b7280")
        self.status.pack(anchor="w")
        tk.Label(right, text="가능성이 높은 후보", font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(8, 2))
        self.cands = tk.Canvas(right, width=260, height=ROW_H * TOP_K, highlightthickness=0)
        self.cands.pack(anchor="w")

        tk.Label(right, text="라벨을 눌러 저장 (키보드 0–9)").pack(anchor="w", pady=(14, 4))
        grid = tk.Frame(right)
        grid.pack(anchor="w")
        for d in range(10):
            tk.Button(grid, text=str(d), width=4, font=("Segoe UI", 12, "bold"),
                      command=lambda d=d: self.save_sample(d)
                      ).grid(row=d // 5, column=d % 5, padx=2, pady=2)

        self.stats = tk.Label(right, justify="left", fg="#555")
        self.stats.pack(anchor="w", pady=(12, 0))
        tk.Label(right, text=f"저장 위치: {CSV_PATH.name}", fg="#888").pack(anchor="w")

        # ---- 단축키 ----
        root.bind("<Escape>", lambda e: self.clear())
        root.bind("<Control-z>", lambda e: self.undo())
        for d in range(10):
            root.bind(str(d), lambda e, d=d: self.save_sample(d))

        self.redraw()
        self.recognize()
        self.update_stats()

    # ---------- 그리기 ----------
    def on_down(self, e):
        self.current = {"w": self.brush.get(), "pts": [(e.x, e.y)]}
        self.strokes.append(self.current)
        self.redraw()

    def on_move(self, e):
        if self.current:
            self.current["pts"].append((e.x, e.y))
            self.redraw()

    def on_up(self, e):
        self.current = None
        self.recognize()

    def render_image(self) -> Image.Image:
        """획 목록을 280x280 흑백 PIL 이미지로 그린다."""
        img = Image.new("L", (SIZE, SIZE), 0)
        d = ImageDraw.Draw(img)
        for s in self.strokes:
            w, pts = s["w"], s["pts"]
            r = w / 2
            if len(pts) > 1:
                d.line(pts, fill=255, width=w)
            for x, y in pts:                     # 둥근 끝/이음새
                d.ellipse((x - r, y - r, x + r, y + r), fill=255)
        return img

    def redraw(self):
        self.image = self.render_image()
        self.tk_canvas_img = ImageTk.PhotoImage(self.image)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_canvas_img)

        self.pixels = to_mnist(np.asarray(self.image))
        pixels = self.pixels
        prev = Image.fromarray(pixels if pixels is not None else np.zeros((28, 28), np.uint8))
        self.tk_preview = ImageTk.PhotoImage(prev.resize((PREVIEW, PREVIEW), Image.NEAREST))
        self.preview.config(image=self.tk_preview)

    def clear(self):
        self.strokes = []
        self.redraw()
        self.recognize()

    def undo(self):
        if self.strokes:
            self.strokes.pop()
            self.redraw()
            self.recognize()

    # ---------- 인식 ----------
    def load_model(self) -> MLP | None:
        # 없거나 깨진 파일이어도 수집 기능은 계속 쓸 수 있어야 한다.
        try:
            return MLP.load(MODEL_PATH)
        except Exception:
            return None

    def show_candidates(self, ranked: list[tuple[int, float]]):
        """후보 줄을 그린다. ranked 가 비었으면 '-' 만 표시."""
        c = self.cands
        c.delete("all")
        for k in range(TOP_K):
            y = k * ROW_H + ROW_H / 2
            c.create_rectangle(24, y - 7, 24 + BAR_W, y + 7, fill="#eceef2", width=0)
            if k < len(ranked):
                d, p = ranked[k]
                c.create_text(4, y, text=str(d), anchor="w", font=("Segoe UI", 11))
                c.create_rectangle(24, y - 7, 24 + BAR_W * max(0.01, p), y + 7, fill="#2563eb", width=0)
                c.create_text(32 + BAR_W, y, text=f"{p * 100:.1f}%", anchor="w", fill="#6b7280")
            else:
                c.create_text(4, y, text="-", anchor="w", font=("Segoe UI", 11))

    def recognize(self, from_button: bool = False):
        if self.model is None:
            self.digit.config(text="?")
            self.status.config(text="모델 없음 — train.bat을 먼저 실행하세요")
            self.show_candidates([])
            return
        if self.pixels is None:
            self.digit.config(text="?")
            self.status.config(text="아직 그린 내용이 없습니다." if from_button else HINT)
            self.show_candidates([])
            return
        probs = self.model.predict_proba(self.pixels.reshape(784) / 255)
        ranked = [(int(d), float(probs[d])) for d in np.argsort(-probs)[:TOP_K]]
        self.digit.config(text=str(ranked[0][0]))
        self.status.config(text=f"확신도 {ranked[0][1] * 100:.1f}%")
        self.show_candidates(ranked)

    # ---------- 저장 ----------
    def load_counts(self) -> list[int]:
        counts = [0] * 10
        if CSV_PATH.exists():
            with CSV_PATH.open(newline="") as f:
                reader = csv.reader(f)
                next(reader, None)               # 헤더 건너뛰기
                for row in reader:
                    if row:
                        counts[int(row[0])] += 1
        return counts

    def save_sample(self, label: int):
        pixels = self.pixels
        if pixels is None:
            return
        new_file = not CSV_PATH.exists()
        with CSV_PATH.open("a", newline="") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(["label"] + [f"pixel{i}" for i in range(784)])
            w.writerow([label] + pixels.flatten().tolist())
        self.counts[label] += 1
        self.update_stats()
        self.clear()

    def update_stats(self):
        total = sum(self.counts)
        per = "  ".join(f"{d}:{c}" for d, c in enumerate(self.counts))
        self.stats.config(text=f"총 {total}개 저장됨\n{per}")


if __name__ == "__main__":
    root = tk.Tk()
    try:
        App(root)
    except Exception as ex:                      # 더블클릭 실행 시 에러가 보이도록
        messagebox.showerror("오류", str(ex))
        raise
    root.mainloop()
