# 손글씨 숫자 인식기 (웹 / 데스크톱) 설계

- 작성일: 2026-09-27
- 브랜치: `feature/recognizer`

## 1. 목표

지금 있는 손글씨 **수집기**를 **인식 기능이 있는 프로그램**으로 확장하고, 웹 버전과 데스크톱 버전으로 나눈다.

| 버전 | 역할 | 제약 |
|---|---|---|
| `desktop_version/` | 기존 수집 기능 + 실시간 인식, 모델 학습 | Python 3.14 + numpy + Pillow만 사용 (새 라이브러리 설치 없음) |
| `web_version/` | 인식 전용 데모 | 외부 라이브러리 없는 순수 JS, GitHub Pages에 정적 배포, `file://`로 더블클릭해 열어도 동작 |

### 성공 기준

1. 공식 MNIST 시험 데이터(10,000장)에서 test 정확도 **97% 이상** (파이썬)
2. 검증 데이터 200장 기준으로 JS와 파이썬 결과가 일치 (세부 기준은 §7)
3. 데스크톱 앱과 웹 페이지에서 직접 그린 숫자가 실시간으로 인식됨
4. CPU로 학습이 몇 분 안에 끝남

### 하지 않는 것 (YAGNI)

- CNN, 데이터 증강, 드롭아웃, 학습률 스케줄 (MLP로 97%를 넘기지 못하거나 직접 그린 글씨 인식이 나쁠 때 다시 검토)
- 웹 버전의 수집·저장 기능
- 웹 폴더와 데스크톱 폴더 사이의 코드 공유 (각 폴더는 독립)

## 2. 폴더 구조

```
study01_MNIST/
├─ CLAUDE.md                     전체 개요 + 공통 데이터 계약 (갱신)
├─ .gitignore                    desktop_version/data/ 추가
├─ .github/workflows/pages.yml   web_version/만 GitHub Pages로 배포
├─ docs/superpowers/             스펙·계획 문서
├─ desktop_version/
│  ├─ CLAUDE.md
│  ├─ app.py            (git mv) 수집기 + 인식 패널 추가
│  ├─ index.html        (git mv) 브라우저 수집기, toMnist만 면적 평균으로 동기화
│  ├─ run.bat           (git mv)
│  ├─ train.bat         신규: 더블클릭으로 학습
│  ├─ preprocess.py     신규: to_mnist / area_downscale (app.py에서 분리, tkinter 의존 없음)
│  ├─ mnist_model.py    신규: numpy MLP
│  ├─ mnist_data.py     신규: 공식 MNIST 다운로드·파싱, my_mnist.csv 로드
│  ├─ train.py          신규: 학습 → model.npz + ../web_version/model.js
│  ├─ make_golden.py    신규: 검증 데이터 → ../web_version/golden.js
│  ├─ model.npz         생성물 (커밋)
│  ├─ my_mnist.csv      수집 데이터 (지금처럼 app.py 옆)
│  ├─ data/             MNIST 원본 캐시 (git 제외)
│  └─ tests/            unittest (python -m unittest)
└─ web_version/
   ├─ CLAUDE.md
   ├─ index.html        그리기 + 28×28 미리보기 + 확률 막대
   ├─ preprocess.js     toMnist / areaDownscale (DOM 비의존, 픽셀 배열 입력)
   ├─ mlp.js            base64 디코딩, 순전파, softmax
   ├─ model.js          생성물: window.MNIST_MODEL
   ├─ golden.js         생성물: window.MNIST_GOLDEN (커밋)
   └─ tests.html        브라우저 테스트 페이지
```

파일 이름은 영문으로 한다. 주석과 UI 문자열은 한국어로 쓴다.

## 3. 공통 데이터 계약

- **샘플**: 28×28, 검은 배경(0)에 흰 획(255), uint8, 행 우선 784개
- **CSV**: 헤더 `label,pixel0,...,pixel783` (Kaggle `mnist_train.csv`와 같음)
- **모델 입력**: `pixels / 255` (float, 0–1)

## 4. 전처리 (세 곳에서 같은 알고리즘)

구현 위치: `desktop_version/preprocess.py`, `desktop_version/index.html`, `web_version/preprocess.js`.
하나를 바꾸면 나머지 둘도 같이 바꾼다.

입력은 280×280 흑백 비트맵(0–255)이다.

1. **경계 상자**: 값이 20보다 큰 픽셀의 경계 상자를 구한다. 해당 픽셀이 없으면 `null`/`None`을 돌려준다.
2. **목표 크기**: `scale = 20 / max(bw, bh)`, `sw = max(1, round(bw*scale))`, `sh = max(1, round(bh*scale))`.
   `round`는 `floor(v + 0.5)`로 한다. 파이썬 `round`와 `np.round`는 짝수 반올림(banker's rounding)이라 쓰지 않는다.
3. **면적 평균 축소**: 출력 픽셀 `(ox, oy)`는 소스 구간 `[ox*bw/sw, (ox+1)*bw/sw) × [oy*bh/sh, (oy+1)*bh/sh)`를 덮는다.
   소스 픽셀마다 겹친 면적을 가중치로 곱해 더하고, 덮은 면적으로 나눈다. 결과는 `clip(floor(v + 0.5), 0, 255)`로 정수화한다.
   가로와 세로를 나눠 계산하는 분리형(1D 가중치 행렬 곱 두 번)으로 구현해도 결과는 같다.
   PIL의 LANCZOS와 `Image.BOX`는 쓰지 않는다. `Image.BOX`는 픽셀 중심이 구간에 포함되는지로 판정해서 정확한 면적 평균과 다르다.
4. **무게중심 이동**: 정수화된 축소 결과로 무게중심 `(cx, cy)`를 구한다.
   이동량은 `ox = floor(14 - cx)`, `oy = floor(14 - cy)`이다. 기존 식 `round(14 - c - 0.5)`를 `floor(v + 0.5)`로 반올림한 결과와 같다. 축소 결과의 `(x, y)`를 28×28의 `(x+ox, y+oy)`에 놓고, 범위를 벗어난 픽셀은 버린다.

이 방식은 기존 앱의 전처리(LANCZOS / 브라우저 `drawImage`)와 결과가 약간 다르다. 지금 `my_mnist.csv` 샘플이 0개라 영향은 없다.

## 5. 모델 (`mnist_model.py`)

- 구조: 784 → 128 → 64 → 10. 은닉층은 ReLU, 출력층은 softmax, 손실은 교차 엔트로피
- 가중치: `W`는 `(in, out)` 모양 float32, `y = x @ W + b`. JS도 같은 행 우선 규칙을 쓴다
- 초기화는 He 초기화. 최적화는 Adam(lr 1e-3, β1 0.9, β2 0.999, ε 1e-8), 배치 64, 10 에포크, 시드 고정
- 공개 API
  - `MLP(sizes=(784,128,64,10), seed=0)`
  - `forward(x) -> probs`, `loss_and_grads(x, y) -> (loss, grads)`, `train_step(x, y) -> loss`
  - `predict_proba(x)` (1개 또는 배치)
  - `save(path)` / `MLP.load(path)` (npz)
  - `export_js(path, meta)` → `window.MNIST_MODEL = {...};`

### model.js 형식

```js
window.MNIST_MODEL = {
  version: 1,
  testAccuracy: 0.978,
  layers: [
    { inSize: 784, outSize: 128, W: "<base64 float32 little-endian>", b: "<base64>" },
    { inSize: 128, outSize: 64,  W: "...", b: "..." },
    { inSize: 64,  outSize: 10,  W: "...", b: "..." }
  ]
};
```

은닉층은 ReLU, 마지막 층은 softmax로 고정한다. 크기는 약 580KB이다.

## 6. 데이터와 학습

- `mnist_data.load_mnist(cache_dir)`: `data/`에 `train-images-idx3-ubyte.gz` 등 4개 파일이 없으면 다운로드한다.
  첫 번째 미러는 `https://storage.googleapis.com/cvdf-datasets/mnist/`, 실패하면 `https://ossci-datasets.s3.amazonaws.com/mnist/`를 쓴다.
  두 곳 다 실패하면 수동으로 받아 `data/`에 넣으라는 안내를 출력하고 종료한다. IDX 파서는 매직 넘버를 검사한다.
- `mnist_data.load_my_csv(path)`: 파일이 없으면 빈 배열을 돌려주고, 있으면 `(X uint8 (n,784), y)`를 돌려준다.
- `train.py`
  1. 공식 train 60k에 my_mnist.csv를 더한다.
  2. 시드를 고정해 섞은 뒤 5,000개를 검증용으로 분리한다.
  3. 에포크마다 `epoch 3/10  loss 0.081  val 97.6%`를 출력한다.
  4. 공식 test 10k 정확도를 출력한다. 97% 미만이면 경고한다.
  5. `model.npz`와 `../web_version/model.js`를 쓴다.
- `train.bat`: `run.bat`과 같은 방식으로 실제 인터프리터(`python.exe`, 콘솔 있음)를 써서 `train.py`를 실행하고, 끝나면 `pause`한다.

## 7. 검증 데이터와 검증 (TDD: 검사 수단을 먼저 만든다)

### 검증 데이터 (`make_golden.py` → `web_version/golden.js`)

- 공식 MNIST 시험 이미지 200장을 고정 시드로 고른다.
- 앱 입력과 같은 조건을 만들려고 각 이미지를 **최근접 이웃 ×10**으로 280×280까지 키운다. `big[y][x] = img[floor(y/10)][floor(x/10)]`이고, 파이썬과 JS가 같은 규칙을 쓴다.
- 이미지마다 저장하는 값
  - `label`
  - `src`: 28×28 원본 (base64 uint8)
  - `pre`: 파이썬 전처리 결과 (base64 uint8)
  - `probs`: 파이썬 모델이 `pre`로 계산한 확률 10개
- 280×280 이미지는 저장하지 않는다. 28×28 원본만 저장하고 키우는 건 검사 때 한다. 그래서 파일이 약 450KB이고 커밋할 수 있다.
- `window.MNIST_GOLDEN = {...}` 형태의 `.js`로 만든다. `file://`에서는 `.json`을 `fetch`로 읽을 수 없기 때문이다.

### 웹 검증 (`tests.html`)

| 검사 | 입력 | 기준 |
|---|---|---|
| base64 디코딩 | 알려진 바이트열 | 정확히 일치 |
| 순전파 | `pre` → JS MLP | `probs`와 최대 절대 오차 < 1e-4 |
| 전처리 | `src` → ×10 → JS toMnist | `pre`와 완전 일치 ≥ 198/200. 불일치 이미지는 번호와 최대 픽셀 차를 표시 (부동소수점 합산 순서 때문에 0.5 경계에서 1px 밀릴 수 있음) |
| 전체 경로 | `src` → ×10 → JS toMnist → JS MLP | 파이썬 예측(argmax)과 일치 ≥ 199/200, JS 정확도 출력 |
| 빈 입력 | 검은 280×280 | `null` |

페이지 맨 위에 통과/실패 요약을 보여준다.

### 파이썬 검증 (`desktop_version/tests/`, unittest)

- `test_model.py`: 수치 기울기 검사, 작은 합성 데이터에서 손실 감소, save/load 왕복, `export_js`의 base64를 디코딩하면 원래 가중치와 같음
- `test_data.py`: 합성 IDX 바이트 파싱, 잘못된 매직 넘버 거부, `load_my_csv`(파일 없음 / 정상)
- `test_preprocess.py`: 면적 평균 축소를 손으로 계산한 값과 비교, 빈 이미지는 None, 결과의 무게중심이 (14,14) 근처, `floor(v+0.5)` 반올림
- 네트워크 다운로드는 테스트하지 않는다.

## 8. UI

### 데스크톱 `app.py`

- 오른쪽 패널에서 미리보기 아래에 "인식 결과" 영역을 추가한다. 예측 숫자를 크게 보여주고 0–9 확률은 `tk.Canvas` 가로 막대로 그린다.
- `redraw()`마다 이미 계산한 28×28로 추론한다.
- `model.npz`가 없으면 "모델 없음 — train.bat을 먼저 실행하세요"를 표시한다. 수집 기능은 그대로 동작한다.
- `to_mnist`는 `preprocess.py`에서 가져온다.

### 웹 `index.html`

- 기존 수집기의 스타일(CSS 변수, 다크 모드)을 따른다. 그리기 캔버스, 28×28 미리보기, 예측 숫자, 확률 막대 10개로 구성한다.
- 지우기(Esc)와 되돌리기(Ctrl+Z)를 지원하고, 포인터 이벤트로 마우스와 터치를 모두 받는다. 휴대폰 폭에서는 세로로 쌓는다.
- `window.MNIST_MODEL`이 없거나 가중치 길이가 맞지 않으면 화면에 오류를 표시한다. 캔버스가 비었으면 "숫자를 그려 보세요"를 보여준다.
- 스크립트는 `model.js`, `mlp.js`, `preprocess.js` 순서의 `<script src>`로 불러온다. 모두 상대 경로이다.

## 9. 배포

`.github/workflows/pages.yml`은 `main`에 push하면 `actions/upload-pages-artifact`로 `web_version/`만 올린 뒤 `actions/deploy-pages`로 배포한다.
사용자는 저장소 Settings → Pages → Source를 **GitHub Actions**로 한 번만 바꾸면 된다. push는 사용자 확인을 받은 뒤에 한다.

## 10. 이전 작업 시 주의

- `git mv`로 파일을 옮겨 히스토리를 유지한다.
- 브라우저 수집기의 샘플은 `localStorage`에 있다. 파일 위치가 바뀌면 브라우저(특히 Firefox)에 따라 안 보일 수 있다. 이미 모은 샘플이 있다면 이동 전에 "CSV 내보내기"로 백업한다.
- 수집한 데이터는 자동으로 반영되지 않는다. `train.bat`을 다시 실행해야 두 버전의 모델이 모두 갱신된다. 웹은 `model.js`를 커밋하고 push해야 반영된다.

## 11. CLAUDE.md

- 루트: 두 폴더 개요, 공통 데이터 계약, 전처리를 세 곳에서 동기화하는 규칙, 재학습 흐름
- `desktop_version/CLAUDE.md`: 실행(run.bat / train.bat), 테스트 명령, 모듈 역할, Windows 인터프리터 경로
- `web_version/CLAUDE.md`: 외부 라이브러리 금지, `file://` 호환(`fetch` 금지, `.js`로 데이터 로드), 생성물(`model.js`, `golden.js`)은 손으로 고치지 않음, `tests.html`로 검증, 배포 방식
