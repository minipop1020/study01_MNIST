# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

MNIST study project. Currently a handwritten-digit input tool that collects user-drawn digits as MNIST-format samples (28×28 grayscale, black background, white strokes 0–255). The user communicates in Korean; UI strings and comments are Korean.

Two independent implementations of the same tool:
- `app.py` — tkinter desktop app (numpy + Pillow). Appends each sample to `my_mnist.csv` next to the script.
- `index.html` — standalone browser version, no dependencies. Stores samples in `localStorage` (key `mnist-handwriting-samples`) and exports CSV/JSON on demand.

Both share the same data contract and preprocessing — **keep them in sync when changing either**:
- CSV format: header `label,pixel0,...,pixel783`, one row per sample (same as Kaggle `mnist_train.csv`).
- Preprocessing (`to_mnist` in `app.py`, `toMnist` in `index.html`): crop to the stroke bounding box (threshold > 20), resize preserving aspect ratio so the longer side is 20px, then place in a 28×28 frame so the center of mass lands at (14,14). This mirrors how the original MNIST was built.
- Both apps keep a list of strokes and re-render the 280×280 canvas from it (enables undo); the preview is recomputed on every redraw.

## Running

- Desktop app: double-click `run.bat` (or `python app.py`). `.py` files are **not** associated with Python on this machine, so double-clicking `app.py` does not work — `run.bat` launches it with `pythonw.exe` directly.
- Browser version: open `index.html` in a browser.

There are no tests, build step, or linter.

## Environment notes (Windows)

- Python 3.14 is installed via the Python install manager. The real interpreter is `%LOCALAPPDATA%\Python\pythoncore-3.14-64\python.exe` (`pythonw.exe` for no console). `run.bat` hardcodes this path — update it if Python is upgraded.
- The `python`/`py` commands on PATH are WindowsApps aliases; they have been observed failing to see some folders (e.g. under `AppData\Roaming`). If `python` reports missing files/modules for paths that exist, call the real interpreter by full path.
- numpy and Pillow were installed with `pip install --user`.
- The project lives in OneDrive-synced Documents (`C:\Users\jimin\OneDrive\문서\study01_MNIST`); the path contains Korean characters, so quote paths.
