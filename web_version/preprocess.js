// MNIST 방식 전처리 (DOM 비의존, 픽셀 배열 입력)
//
// 같은 알고리즘이 세 곳에 있다. 하나를 바꾸면 나머지도 같이 바꾼다.
//   - desktop_version/preprocess.py
//   - desktop_version/index.html   toMnist
//   - web_version/preprocess.js    (이 파일)
//
// 1. 값 > 20 인 픽셀의 경계 상자를 구한다.
// 2. 긴 변이 20px 이 되도록 비율을 유지해 면적 평균으로 축소한다.
// 3. 무게중심이 28×28 의 (14,14) 에 오도록 놓는다.
// 반올림은 항상 floor(v + 0.5) 이다 (Math.round 는 음수에서 다르게 동작한다).
window.MnistPre = (() => {
  const roundHalfUp = v => Math.floor(v + 0.5);

  // 출력 칸 o 가 소스 구간 [o*n/m, (o+1)*n/m) 과 겹치는 길이. 0 이 아닌 것만 [i, w] 목록으로.
  function axisWeights(nSrc, nOut) {
    const step = nSrc / nOut, all = [];
    for (let o = 0; o < nOut; o++) {
      const lo = o * step, hi = lo + step, list = [];
      for (let i = Math.floor(lo); i < Math.min(nSrc, Math.ceil(hi)); i++) {
        const w = Math.min(i + 1, hi) - Math.max(i, lo);
        if (w > 0) list.push([i, w]);
      }
      all.push(list);
    }
    return all;
  }

  // src(너비 srcW)의 (x0, y0, bw, bh) 영역을 sw×sh 로 정확한 면적 평균 축소. 가로 먼저, 세로 다음.
  function areaDownscale(src, srcW, x0, y0, bw, bh, sw, sh) {
    const wx = axisWeights(bw, sw), wy = axisWeights(bh, sh);
    const rows = new Float64Array(bh * sw);
    for (let y = 0; y < bh; y++) {
      const base = (y0 + y) * srcW + x0;
      for (let ox = 0; ox < sw; ox++) {
        let s = 0;
        for (const [i, w] of wx[ox]) s += src[base + i] * w;
        rows[y * sw + ox] = s;
      }
    }
    const area = (bw / sw) * (bh / sh);
    const out = new Uint8Array(sw * sh);
    for (let oy = 0; oy < sh; oy++) {
      for (let ox = 0; ox < sw; ox++) {
        let s = 0;
        for (const [i, w] of wy[oy]) s += rows[i * sw + ox] * w;
        out[oy * sw + ox] = Math.min(255, Math.max(0, roundHalfUp(s / area)));
      }
    }
    return out;
  }

  // W×H 흑백 배열 → 28×28 Uint8Array(784). 그린 것이 없으면 null.
  function toMnist(gray, W, H) {
    let minX = W, minY = H, maxX = -1, maxY = -1;
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        if (gray[y * W + x] > 20) {
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
        }
      }
    }
    if (maxX < 0) return null;

    const bw = maxX - minX + 1, bh = maxY - minY + 1;
    const scale = 20 / Math.max(bw, bh);
    const sw = Math.max(1, roundHalfUp(bw * scale)), sh = Math.max(1, roundHalfUp(bh * scale));
    const small = areaDownscale(gray, W, minX, minY, bw, bh, sw, sh);

    // 정수화된 축소 결과로 무게중심을 구해 (14,14) 로 옮긴다.
    let sum = 0, cx = 0, cy = 0;
    for (let y = 0; y < sh; y++) {
      for (let x = 0; x < sw; x++) {
        const v = small[y * sw + x];
        sum += v; cx += x * v; cy += y * v;
      }
    }
    if (sum === 0) return null;                   // 축소하며 모두 0 이 된 경우
    const ox = Math.floor(14 - cx / sum), oy = Math.floor(14 - cy / sum);

    const out = new Uint8Array(784);
    for (let y = 0; y < sh; y++) {
      const ty = y + oy;
      if (ty < 0 || ty >= 28) continue;
      for (let x = 0; x < sw; x++) {
        const tx = x + ox;
        if (tx >= 0 && tx < 28) out[ty * 28 + tx] = small[y * sw + x];
      }
    }
    return out;
  }

  // 28×28 → 280×280 최근접 이웃 확대. big[y][x] = img[y/10][x/10] (검사용)
  function upscale10(src28) {
    const big = new Uint8Array(280 * 280);
    for (let y = 0; y < 280; y++) {
      for (let x = 0; x < 280; x++) big[y * 280 + x] = src28[Math.floor(y / 10) * 28 + Math.floor(x / 10)];
    }
    return big;
  }

  // 캔버스 ImageData → 흑백 배열 (R 채널만 쓴다. 흰 획이라 R=G=B)
  function grayFromImageData(imageData) {
    const { data, width, height } = imageData;
    const gray = new Uint8Array(width * height);
    for (let i = 0; i < gray.length; i++) gray[i] = data[i * 4];
    return gray;
  }

  return { roundHalfUp, areaDownscale, toMnist, upscale10, grayFromImageData };
})();
