# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

MNIST study project: a handwritten-digit collector + recognizer. The user communicates in Korean; UI strings and comments are Korean, file names are English.

Two independent folders (no code shared between them):
- `desktop_version/` — tkinter app that collects samples into `my_mnist.csv` **and** recognizes digits live; also holds the numpy MLP, training script, and tests. See `desktop_version/CLAUDE.md`.
- `web_version/` — recognition-only demo in plain JS, deployed to GitHub Pages. See `web_version/CLAUDE.md`.

Design spec and implementation plan: `docs/superpowers/specs/`, `docs/superpowers/plans/`.

## Shared data contract

- Sample: 28×28, black background (0), white strokes (255), uint8, row-major 784 values.
- CSV: header `label,pixel0,...,pixel783` (same as Kaggle `mnist_train.csv`).
- Model input: `pixels / 255`. Model: 784 → 128 → 64 → 10, ReLU hidden, softmax output, `W` shaped `(in, out)` so `y = x @ W + b`.

## Preprocessing lives in three places — keep them in sync

`desktop_version/preprocess.py`, `desktop_version/index.html` (`toMnist`), `web_version/preprocess.js`. Same algorithm, change all three together:
1. Bounding box of pixels > 20 (none → `None`/`null`).
2. `scale = 20 / max(bw, bh)`, target size `max(1, floor(v*scale + 0.5))`; **exact area-average** downscale (not PIL LANCZOS/BOX, not canvas `drawImage`), result `clip(floor(v + 0.5))`.
3. Center of mass of the integer result; shift `floor(14 - c)`; drop pixels outside 28×28.
Rounding is always `floor(v + 0.5)` — never Python `round`/`np.round` (banker's rounding) or JS `Math.round`.

`web_version/tests.html` checks the JS side against Python-generated golden data, so run it after touching any of the three.

## Retraining flow

1. Collect samples (desktop app → `my_mnist.csv`; browser collector → export CSV and merge into it).
2. `desktop_version/train.bat` → writes `desktop_version/model.npz` and `web_version/model.js` (test accuracy printed; warns below 97%).
3. `python make_golden.py` (in `desktop_version/`) → regenerates `web_version/golden.js`; open `web_version/tests.html` to confirm.
4. Commit `model.npz`, `model.js`, `golden.js`; pushing to `main` redeploys the web version.

## Deployment

`.github/workflows/pages.yml` uploads only `web_version/` to GitHub Pages on push to `main`. One-time setup: repo Settings → Pages → Source = **GitHub Actions**. Push only after the user confirms.

## Environment notes (Windows)

- Python 3.14 via the Python install manager. Real interpreter: `%LOCALAPPDATA%\Python\pythoncore-3.14-64\python.exe` (`pythonw.exe` for no console). `run.bat`/`train.bat` hardcode this path — update them if Python is upgraded.
- The `python`/`py` commands on PATH are WindowsApps aliases and have been seen failing on some folders; call the real interpreter by full path if they misbehave.
- numpy and Pillow were installed with `pip install --user`. No other packages — don't add any.
- `.py` files are not associated with Python; double-clicking `app.py` does nothing. Use the `.bat` files.
- The project lives in OneDrive-synced Documents (`C:\Users\jimin\OneDrive\문서\study01_MNIST`); the path contains Korean characters, so quote paths.
