# desktop_version

Collector + live recognizer (tkinter) and everything that produces the model. Only numpy, Pillow and the standard library.

## Running

- App: double-click `run.bat` (launches `app.py` with `pythonw.exe`).
- Train: double-click `train.bat` (runs `train.py` with console `python.exe`, then pauses). First run downloads MNIST into `data/` (git-ignored); if both mirrors fail it prints how to place the files manually.
- Golden data for the web tests: `python make_golden.py` (after retraining).
- Tests: from this folder, `"%LOCALAPPDATA%\Python\pythoncore-3.14-64\python.exe" -m unittest discover -s tests -v`. Network download is not tested.

## Modules

- `app.py` — tkinter UI. Keeps a list of strokes and re-renders the 280×280 canvas from it (enables undo); `redraw()` recomputes the 28×28 (`self.pixels`) and preview. Recognition runs on stroke release, clear, undo, and the 인식하기 button, not during drag. Shows `?` + hint when empty, "모델 없음 — train.bat을 먼저 실행하세요" if `model.npz` is missing (collection still works). Label buttons / keys 0–9 append to `my_mnist.csv`.
- `preprocess.py` — `to_mnist`, `area_downscale`, `round_half_up`, `upscale10`. No tkinter. Must match `index.html` and `../web_version/preprocess.js` (see root CLAUDE.md).
- `mnist_model.py` — `MLP` (He init, Adam, `save`/`load` npz, `export_js` → `window.MNIST_MODEL` with base64 float32 LE weights).
- `mnist_data.py` — `parse_idx`, `load_mnist(cache_dir)`, `load_my_csv(path)`.
- `augment.py` — `augment(X, rng, y)`: on half of each training batch, random rotation (±15°)/shear (±0.3)/anisotropic scale/shift (bilinear) and stroke thickening/thinning; `add_flag` gives 35% of the 1s a top-left flag stroke. Without it the model misread mouse-drawn digits (a 6 with a big loop → 0, a flagged 1 → 7). Settings were chosen by drawing synthetic mouse-style digits in the web page (≈99.5% vs 97.2% before).
- `train.py` — official 60k + `my_mnist.csv`, 5k validation split, MLP `SIZES` 784→256→128→10, 30 epochs with augmented batches, lr ×0.93 per epoch, batch 64, fixed seeds → `model.npz` + `../web_version/model.js`. Takes about 4 minutes (test 99.1%).
- `make_golden.py` — 200 fixed test images → `../web_version/golden.js`.
- `index.html` — standalone browser collector (localStorage key `mnist-handwriting-samples`, CSV/JSON export). Its `toMnist` is a copy of the shared algorithm.

Generated files (`model.npz`, `../web_version/model.js`, `../web_version/golden.js`) are committed; regenerate rather than hand-edit.
