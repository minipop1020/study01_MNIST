# 손글씨 숫자 인식기 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 기존 수집기를 `desktop_version/`(수집 + 실시간 인식 + 학습)과 `web_version/`(인식 전용 데모, GitHub Pages)으로 나누고, numpy MLP로 학습한 모델로 직접 그린 숫자를 인식한다.

**Architecture:** 파이썬(numpy)에서 MLP를 학습해 `model.npz`와 `web_version/model.js`(base64 가중치)를 만든다. 전처리(면적 평균 축소 + 무게중심 정렬)는 파이썬·브라우저 수집기·웹 인식기 세 곳에 같은 알고리즘으로 두고, 파이썬이 만든 검증 데이터(`golden.js`)로 JS 구현이 같은 결과를 내는지 확인한다.

**Tech Stack:** Python 3.14 + numpy + Pillow + tkinter + unittest / 순수 JS(클래식 `<script>`, 모듈·fetch 없음) / GitHub Actions Pages

**Spec:** `docs/superpowers/specs/2026-09-27-handwriting-recognizer-design.md`

## Global Constraints

- 새 파이썬 라이브러리 설치 금지 (numpy, Pillow, 표준 라이브러리만)
- 웹: 외부 라이브러리 금지, `file://`로 열어도 동작 → `fetch`·ES 모듈 금지, 데이터는 `window.X = {...}` 형태 `.js`로 로드
- 파일 이름은 영문, 주석·UI 문자열은 한국어
- 반올림은 항상 `floor(v + 0.5)` (파이썬 `round`/`np.round` 금지)
- 샘플: 28×28, 검은 배경 0, 흰 획 255, uint8, 행 우선 784개. 모델 입력은 `pixels / 255`
- 모델: 784 → 128 → 64 → 10, ReLU/softmax, He 초기화, Adam(lr 1e-3, β1 0.9, β2 0.999, ε 1e-8), 배치 64, 10 에포크, 시드 고정. `W`는 `(in, out)` float32
- 파이썬 실행은 실제 인터프리터 경로로: `"$LOCALAPPDATA/Python/pythoncore-3.14-64/python.exe"` (아래에서 `PY`라고 부름). 경로에 한글이 있으니 항상 따옴표
- push는 사용자 확인 후에만

## Review Focus

1. **점 하나 / 아주 가는 선** (경계 상자 1×N): `sw`, `sh`가 최소 1이 되어 오류 없이 결과가 나와야 한다 → Task 2, Task 4 테스트
2. **되돌리기로 캔버스가 다시 비었을 때**: 이전 예측이 남지 말고 `?` 상태로 돌아가야 한다 → Task 5, Task 6 수동 확인 항목
3. **모델 파일 없음/손상**: 웹은 오류 문구 + 단추 비활성화, 데스크톱은 "모델 없음" 표시 후 수집 계속 → Task 3(`loadModel` 길이 검사 테스트), Task 5, Task 6
4. **획이 캔버스 밖으로 나갔다 들어옴**: 좌표가 캔버스 밖이어도 오류 없이 캔버스 안 부분만 인식 → Task 5 수동 확인
5. **두 번째 손가락 터치**: 그리는 중 다른 포인터의 `pointerdown`은 무시 → Task 5

---

## 파일 구조

```
desktop_version/
  app.py, index.html, run.bat   (git mv로 이동)
  preprocess.py   round_half_up, area_downscale, to_mnist, upscale10
  mnist_model.py  MLP
  mnist_data.py   parse_idx, load_mnist, load_my_csv
  train.py, train.bat, make_golden.py
  tests/test_preprocess.py, test_model.py, test_data.py
web_version/
  index.html      인식 UI
  preprocess.js   window.MnistPre
  mlp.js          window.Mlp
  model.js, golden.js   (생성물)
  tests.html
.github/workflows/pages.yml
```

---

### Task 1: 폴더 재구성

**Files:**
- Move: `app.py`, `index.html`, `run.bat` → `desktop_version/` (`git mv`)
- Modify: `.gitignore` (`desktop_version/data/` 추가)

- [ ] **Step 1:** `git mv app.py index.html run.bat desktop_version/`, `mkdir web_version desktop_version/tests`
- [ ] **Step 2:** `.gitignore`에 `desktop_version/data/` 추가
- [ ] **Step 3:** 확인 — `desktop_version/run.bat`은 `cd /d "%~dp0"`를 쓰므로 수정 없이 동작. `PY -c "import ast;ast.parse(open('desktop_version/app.py',encoding='utf-8').read())"` 성공
- [ ] **Step 4:** Commit `refactor: move collector into desktop_version/`

### Task 2: 파이썬 전처리 (`preprocess.py`)

**Files:**
- Create: `desktop_version/preprocess.py`, `desktop_version/tests/test_preprocess.py`

**Interfaces:**
- Produces:
  - `round_half_up(v) -> np.ndarray | float` — `np.floor(v + 0.5)`
  - `area_downscale(src: np.ndarray (bh, bw), sw: int, sh: int) -> np.ndarray uint8 (sh, sw)` — 정확한 면적 평균, `clip(floor(v+0.5),0,255)`
  - `to_mnist(arr: np.ndarray uint8 (H, W)) -> np.ndarray uint8 (28, 28) | None`
  - `upscale10(img28: np.ndarray (28,28)) -> np.ndarray uint8 (280,280)` — 최근접 이웃 `big[y][x] = img[y//10][x//10]` (`np.repeat` 두 번)

- [ ] **Step 1: 실패하는 테스트 작성** (`unittest.TestCase`)
  - `test_round_half_up`: `round_half_up(np.array([0.5, 1.5, 2.5, -0.5]))` == `[1, 2, 3, 0]`
  - `test_area_downscale_exact`: 3×3 `[[0,90,180],[90,180,255],[180,255,255]]`를 2×2로 → 손 계산값. 출력 (0,0)은 소스 구간 `[0,1.5)×[0,1.5)`: 가중치 1,0.5,0.5,0.25 → `(0 + 90*.5 + 90*.5 + 180*.25)/2.25 = 60` → 60. 네 값 모두 손으로 계산해 `assert_array_equal`
  - `test_area_downscale_constant`: 전부 200인 7×5 → 3×2로 줄여도 전부 200
  - `test_empty_returns_none`: `to_mnist(np.zeros((280,280),np.uint8))` is None
  - `test_center_of_mass`: 280×280에 (40..120, 150..260) 직사각형 255 → 결과 무게중심이 `|c-14| < 1` (x, y 모두), 결과 합 > 0
  - `test_longer_side_is_20`: 위 직사각형 결과에서 값>0 인 픽셀 경계 상자의 긴 변 == 20
  - `test_single_dot`: (100,100) 한 픽셀만 255 → None 아님, 합 > 0 (Review Focus 1)
  - `test_thin_line`: 1px 폭 세로선(x=50, y=20..250) → None 아님, 경계 폭 ≥ 1
  - `test_upscale10`: `upscale10(img)[y, x] == img[y//10, x//10]` 무작위 좌표 몇 개
- [ ] **Step 2:** `cd desktop_version && PY -m unittest tests.test_preprocess -v` → ImportError로 FAIL
- [ ] **Step 3: 구현.** 스펙 §4 그대로. `area_downscale`은 분리형: 가로 가중치 행렬 `Wx (sw, bw)`, 세로 `Wy (sh, bh)`를 만들고 `Wy @ src @ Wx.T`를 덮은 면적(`(bw/sw)*(bh/sh)`)으로 나눈다. 가중치 `Wx[o, i] = max(0, min(i+1, (o+1)*bw/sw) - max(i, o*bw/sw))`. 무게중심은 정수화된 축소 결과(float64로)에서 계산, `ox = floor(14 - cx)`, `oy = floor(14 - cy)`, 벗어난 픽셀은 버림
- [ ] **Step 4:** 같은 명령 → 전부 PASS
- [ ] **Step 5:** `app.py`의 `to_mnist`를 지우고 `from preprocess import to_mnist`; 호출부는 `to_mnist(np.asarray(self.image))`로 바꾼다. 모듈 docstring 실행 안내를 `run.bat`으로 수정
- [ ] **Step 6:** Commit `feat: exact area-average preprocessing shared by desktop app`

### Task 3: MLP 모델 (`mnist_model.py`)

**Files:**
- Create: `desktop_version/mnist_model.py`, `desktop_version/tests/test_model.py`

**Interfaces:**
- Produces: `MLP(sizes=(784,128,64,10), seed=0)` — 속성 `Ws: list[np.ndarray float32 (in,out)]`, `bs: list[np.ndarray float32 (out,)]`
  - `forward(x: (n,784) float) -> probs (n,10)`
  - `loss_and_grads(x, y: int array (n,)) -> (loss: float, grads: list[(dW, db)])`
  - `train_step(x, y) -> float` (Adam 한 번)
  - `predict_proba(x) -> (10,)` 1개 입력 (784,) 또는 `(n,10)` 배치
  - `save(path)`, `MLP.load(path) -> MLP` (npz: `W0,b0,W1,b1,W2,b2`)
  - `export_js(path, meta: dict)` → 스펙 §5 형식 `window.MNIST_MODEL = {...};`, W/b는 float32 little-endian base64, `meta`의 키(예: `testAccuracy`)를 최상위에 병합, `version: 1`

- [ ] **Step 1: 실패하는 테스트 작성**
  - `test_gradient_check`: `MLP((5,4,3,2), seed=1)`, x (3,5) 무작위, y `[0,1,1]`. float64로 복사한 가중치로 중앙 차분(eps 1e-5) 수치 기울기 vs 해석 기울기, 상대 오차 < 1e-4 (float32 모델이면 테스트 내에서 가중치를 float64로 바꿔 계산해도 됨 — 구현이 dtype을 따르게 할 것)
  - `test_loss_decreases`: 합성 2클래스 데이터(200개, 784차원, 클래스별 평균 다름) 50 스텝 후 손실 < 처음 손실의 0.5배
  - `test_save_load_roundtrip`: 임시 폴더에 save → load, `predict_proba` 결과 동일
  - `test_export_js_base64`: `export_js` 후 파일에서 `window.MNIST_MODEL = ` 뒤 JSON 파싱(끝의 `;` 제거), 각 층 `W`를 `np.frombuffer(base64.b64decode(..), '<f4').reshape(in,out)` == `Ws[i]`, `inSize/outSize` 맞음, `testAccuracy` 키 존재
  - `test_predict_single_sums_to_one`: `predict_proba(np.zeros(784))` 모양 (10,), 합 ≈ 1
- [ ] **Step 2:** `PY -m unittest tests.test_model -v` → FAIL
- [ ] **Step 3: 구현.** softmax는 최댓값 빼기로 안정화. 교차 엔트로피는 배치 평균. Adam 상태(m, v, t)는 인스턴스에 보관. 난수는 `np.random.default_rng(seed)`
- [ ] **Step 4:** PASS 확인
- [ ] **Step 5:** Commit `feat: numpy MLP with Adam and JS export`

### Task 4: 데이터·학습 (`mnist_data.py`, `train.py`, `train.bat`)

**Files:**
- Create: `desktop_version/mnist_data.py`, `desktop_version/tests/test_data.py`, `desktop_version/train.py`, `desktop_version/train.bat`
- 생성물: `desktop_version/model.npz`, `web_version/model.js`

**Interfaces:**
- Consumes: `MLP` (Task 3)
- Produces:
  - `parse_idx(data: bytes) -> np.ndarray uint8` — 매직 `0x00000803`(이미지, 모양 (n,784)로 reshape) / `0x00000801`(라벨, (n,)). 그 외는 `ValueError`
  - `load_mnist(cache_dir: Path) -> (X_train, y_train, X_test, y_test)` — gzip 4개 파일, 미러 순서와 실패 안내는 스펙 §6
  - `load_my_csv(path: Path) -> (X uint8 (n,784), y int64 (n,))` — 파일 없으면 `(np.zeros((0,784),uint8), np.zeros(0,int64))`

- [ ] **Step 1: 실패하는 테스트 작성**
  - `test_parse_images`: `struct.pack('>IIII', 0x803, 2, 28, 28) + bytes(range(256))*... ` (2×784바이트) → 모양 (2,784), 값 일치
  - `test_parse_labels`: `struct.pack('>II', 0x801, 3) + bytes([7,0,9])` → `[7,0,9]`
  - `test_bad_magic`: `struct.pack('>II', 0x1234, 0)` → `ValueError`
  - `test_my_csv_missing`: 없는 경로 → 모양 (0,784), (0,)
  - `test_my_csv_rows`: 헤더 + 2행 임시 CSV → X 값과 y 일치
- [ ] **Step 2:** `PY -m unittest tests.test_data -v` → FAIL
- [ ] **Step 3:** `mnist_data.py` 구현 (`urllib.request`, `gzip`, `struct`). 다운로드 실패 시 `SystemExit`에 안내 메시지
- [ ] **Step 4:** PASS 확인
- [ ] **Step 5:** `train.py` 작성 — 스펙 §6 순서. 출력 형식 `epoch 3/10  loss 0.081  val 97.6%`, 끝에 `test 97.8%` (97% 미만이면 `경고:` 줄). `model.npz`와 `../web_version/model.js`(`Path(__file__).parent` 기준, `meta={"testAccuracy": round(acc,4)}`)를 쓴다. `train.bat`은 `run.bat`과 같은 구조로 `python.exe train.py` 후 `pause`
- [ ] **Step 6:** `cd desktop_version && PY train.py` 실행 → 10 에포크 출력, test ≥ 97%, 두 파일 생성, `model.js` 약 580KB
- [ ] **Step 7:** Commit (`model.npz`, `web_version/model.js` 포함) `feat: MNIST training script and trained model`

### Task 5: 웹 전처리·추론 + 검증 데이터 + 테스트 페이지

**Files:**
- Create: `web_version/preprocess.js`, `web_version/mlp.js`, `desktop_version/make_golden.py`, `web_version/golden.js`(생성물), `web_version/tests.html`
- Modify: `desktop_version/index.html` (`toMnist`를 같은 알고리즘으로 교체)

**Interfaces:**
- Consumes: `to_mnist`, `upscale10` (Task 2), `MLP.load` (Task 3)
- Produces:
  - `window.MnistPre = { roundHalfUp(v), areaDownscale(src, srcW, x0, y0, bw, bh, sw, sh) -> Uint8Array(sw*sh), toMnist(gray: ArrayLike<number> (W*H), W, H) -> Uint8Array(784) | null, upscale10(src28) -> Uint8Array(78400), grayFromImageData(imageData) -> Uint8Array(W*H) }` — DOM 비의존(마지막 함수만 ImageData의 R 채널을 뽑음)
  - `window.Mlp = { decodeBase64(str) -> Uint8Array, loadModel(spec) -> { testAccuracy, layers: [{inSize, outSize, W: Float32Array, b: Float32Array}] }, predict(model, pixels: ArrayLike (784)) -> Float64Array(10) }` — `loadModel`은 `spec`이 없거나 `W.length !== inSize*outSize`, `b.length !== outSize`, 층 연결 크기 불일치면 `Error`를 던진다. `Float32Array`는 `DataView.getFloat32(i*4, true)`로 읽어 엔디안 고정
  - `window.MNIST_GOLDEN = { items: [{ label, src, pre, probs: number[10] }] }` (src/pre는 base64 uint8 784)

- [ ] **Step 1: `make_golden.py` 작성·실행.** 공식 test 10k에서 `np.random.default_rng(0).choice(10000, 200, replace=False)`, 각 `pre = to_mnist(upscale10(src))`, `probs = model.predict_proba(pre/255)`. `probs`는 소수 7자리로 반올림해 JSON. `PY make_golden.py` → `web_version/golden.js` 생성(약 450KB 이하)
- [ ] **Step 2: 실패하는 테스트 페이지 작성** (`tests.html`: `model.js`, `golden.js`, `mlp.js`, `preprocess.js` 로드 후 검사, 맨 위 요약 `통과 N / 전체 M`, 실패는 빨간 줄). 검사 항목은 스펙 §7 표 그대로 + 다음 두 개:
  - `loadModel` 오류: 첫 층 `W`를 `"AAAA"`로 바꾼 사본 → 예외 발생 (Review Focus 3)
  - 점 하나: 280×280에서 (100,100) 한 픽셀 255 → `toMnist` 결과가 null 아님 (Review Focus 1)
- [ ] **Step 3:** 브라우저 창에서 `file:///.../web_version/tests.html` 열기 → 스크립트 없음으로 실패 확인
- [ ] **Step 4:** `preprocess.js`, `mlp.js` 구현 — 파이썬과 같은 수식. 면적 평균도 분리형(가로 먼저, 세로 다음)으로 파이썬과 합산 순서를 최대한 맞춘다
- [ ] **Step 5:** `tests.html` 새로 고침 → 모두 통과 (전처리 ≥ 198/200, 전체 경로 ≥ 199/200, 순전파 오차 < 1e-4)
- [ ] **Step 6:** `desktop_version/index.html`의 `toMnist()`를 캔버스 R 채널 → 같은 알고리즘(인라인 복사, 폴더 간 공유 금지)으로 교체하고 `drawImage` 축소 코드 제거. 브라우저에서 열어 그림 → 미리보기가 나오는지 확인
- [ ] **Step 7:** Commit `feat: JS preprocessing/inference verified against Python golden data`

### Task 6: 웹 인식 화면 (`web_version/index.html`)

**Files:**
- Create: `web_version/index.html`

**Interfaces:**
- Consumes: `window.MNIST_MODEL`, `Mlp.loadModel/predict`, `MnistPre.toMnist/grayFromImageData` (Task 5)

화면 구성·문구는 스펙 §8 그대로: 왼쪽 안내 "이곳에 숫자를 크게 그려 주세요.", 280×280 검은 캔버스, `지우기`·`인식하기` 단추 / 오른쪽 `인식 결과`, 큰 숫자(약 92px), `확신도 NN.N%`, `가능성이 높은 후보` 3줄(숫자 · 막대 · `NN.N%`, 막대 폭 `max(1, p*100)%`).

- [ ] **Step 1: 구현.**
  - 스크립트 순서: `model.js`, `mlp.js`, `preprocess.js`, 그다음 인라인 앱 스크립트 (모두 상대 경로, 클래식 스크립트)
  - 획 목록(`strokes`)으로 다시 그리는 방식은 기존 수집기와 같게, 붓 굵기 고정 22
  - 포인터: 그리는 중인 `pointerId`만 받는다(두 번째 손가락 무시, Review Focus 5). `pointerup`/`pointercancel`에서 인식 실행
  - `recognize()`: 비었으면 `?` 상태로 초기화(되돌리기로 비었을 때 포함, Review Focus 2), "인식하기" 단추로 호출했고 비었으면 "아직 그린 내용이 없습니다."
  - 지우기(단추, Esc), 되돌리기(Ctrl+Z) 후 `recognize()` 호출
  - 모델 로드 실패 → 확신도 자리에 "모델을 불러오지 못했습니다: <이유>", 단추 `disabled`
  - CSS: 수집기의 변수 체계와 다크 모드, 막대색 `--accent`, 막대 배경 `--border`. `main`은 flex + wrap, 좁은 화면에서 세로로 쌓임. 캔버스 `touch-action: none`, 모드와 무관하게 검은 바탕
- [ ] **Step 2: 확인 (브라우저 창, `file://`)**
  - 7을 그리고 손 떼기 → 큰 숫자 7, 확신도, 후보 3줄
  - Ctrl+Z로 모두 되돌리기 → `?`와 `-`로 돌아감
  - 캔버스 밖으로 끌고 나갔다 들어오기 → 오류 없음(콘솔 확인)
  - 폭 375px → 세로로 쌓이고 가로 스크롤 없음
  - `model.js` 로드를 막은 사본(또는 콘솔에서 `MNIST_MODEL` 삭제 후 초기화 함수 재실행) → 오류 문구와 비활성 단추
  - 0–9 각각 한 번씩 그려서 대부분 맞는지 스크린샷으로 기록
- [ ] **Step 3:** Commit `feat: web recognizer page`

### Task 7: 데스크톱 인식 패널 (`app.py`, `train.bat` 안내)

**Files:**
- Modify: `desktop_version/app.py`

**Interfaces:**
- Consumes: `to_mnist` (Task 2), `MLP.load`, `predict_proba` (Task 3)

- [ ] **Step 1: 구현.**
  - 시작 시 `model.npz`가 있으면 `MLP.load`, 없거나 로드 실패면 `self.model = None`
  - 오른쪽 패널 미리보기 아래에 "인식 결과" 영역: 큰 숫자 `tk.Label`(font 크기 약 60), 확신도 라벨, "가능성이 높은 후보" 라벨, 후보 3줄용 `tk.Canvas`(숫자 · 막대 · 백분율)
  - `recognize()`: `self.pixels`(redraw에서 계산한 28×28)로 추론. `None`이면 `?`/`-` 상태와 "그림을 그리면 자동으로 인식합니다.", 모델 없으면 "모델 없음 — train.bat을 먼저 실행하세요"
  - 호출 시점: `on_up`, `clear`, `undo` 끝(드래그 중에는 부르지 않음). `redraw()`는 `self.pixels = to_mnist(...)`를 저장하고 `save_sample`은 그 값을 재사용
  - "인식하기" 단추를 지우기/되돌리기 줄에 추가
- [ ] **Step 2: 확인** — `run.bat`으로 실행해 숫자를 그리고 결과 확인, 되돌리기로 비우면 `?`, 라벨 저장이 여전히 `my_mnist.csv`에 추가되는지 확인. `model.npz`를 잠시 다른 이름으로 바꾸고 실행 → "모델 없음" 표시, 수집 동작
- [ ] **Step 3:** 전체 파이썬 테스트 `PY -m unittest discover -s tests -v` PASS
- [ ] **Step 4:** Commit `feat: live recognition panel in desktop app`

### Task 8: 배포 워크플로 + 문서

**Files:**
- Create: `.github/workflows/pages.yml`, `desktop_version/CLAUDE.md`, `web_version/CLAUDE.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1:** `pages.yml` — `on: push: branches: [main]` + `workflow_dispatch`, 권한 `pages: write`, `id-token: write`, `contents: read`, `actions/checkout@v4` → `actions/configure-pages@v5` → `actions/upload-pages-artifact@v3` (`path: web_version`) → `actions/deploy-pages@v4`
- [ ] **Step 2:** CLAUDE.md 세 개를 스펙 §11 내용으로 작성/갱신 (전처리 세 곳 동기화 규칙, 재학습 흐름, 테스트 명령, `file://` 제약, 생성물은 손으로 고치지 않음)
- [ ] **Step 3:** Commit `docs: CLAUDE.md for both versions; Pages workflow`
- [ ] **Step 4:** 사용자에게 push·PR 여부와 Settings → Pages → Source를 GitHub Actions로 바꾸는 방법 안내 (push는 확인 후)
