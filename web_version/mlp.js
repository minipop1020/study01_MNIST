// 다층 퍼셉트론 추론 (desktop_version/mnist_model.py 와 같은 규칙)
//
// model.js 의 window.MNIST_MODEL 을 읽는다. 가중치는 float32 little-endian 을 base64 로 담았다.
// W 는 (inSize, outSize) 행 우선이라 y[o] = Σ x[i] * W[i*outSize + o] + b[o] 이다.
// 은닉층은 ReLU, 마지막 층은 softmax.
window.Mlp = (() => {
  function decodeBase64(str) {
    const bin = atob(str);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return bytes;
  }

  function toFloat32(str) {
    const bytes = decodeBase64(str);
    if (bytes.length % 4 !== 0) throw new Error('가중치 바이트 수가 4의 배수가 아닙니다');
    const view = new DataView(bytes.buffer);
    const out = new Float32Array(bytes.length / 4);
    for (let i = 0; i < out.length; i++) out[i] = view.getFloat32(i * 4, true);
    return out;
  }

  // 형식이 맞지 않으면 Error 를 던진다.
  function loadModel(spec) {
    if (!spec || !Array.isArray(spec.layers) || spec.layers.length === 0) {
      throw new Error('model.js 를 찾을 수 없습니다');
    }
    let prevOut = 784;
    const layers = spec.layers.map((l, n) => {
      const W = toFloat32(l.W), b = toFloat32(l.b);
      if (l.inSize !== prevOut) throw new Error(`${n + 1}번째 층 입력 크기가 맞지 않습니다`);
      if (W.length !== l.inSize * l.outSize || b.length !== l.outSize) {
        throw new Error(`${n + 1}번째 층 가중치 길이가 맞지 않습니다`);
      }
      prevOut = l.outSize;
      return { inSize: l.inSize, outSize: l.outSize, W, b };
    });
    if (prevOut !== 10) throw new Error('마지막 층 출력이 10개가 아닙니다');
    return { testAccuracy: spec.testAccuracy, layers };
  }

  // pixels: 0–255 값 784개 → 0–9 확률 10개
  function predict(model, pixels) {
    let x = Float64Array.from(pixels, v => v / 255);
    model.layers.forEach((l, n) => {
      const y = Float64Array.from(l.b);
      for (let i = 0; i < l.inSize; i++) {
        const xi = x[i];
        if (xi === 0) continue;
        const row = i * l.outSize;
        for (let o = 0; o < l.outSize; o++) y[o] += xi * l.W[row + o];
      }
      if (n < model.layers.length - 1) {
        for (let o = 0; o < y.length; o++) if (y[o] < 0) y[o] = 0;
      }
      x = y;
    });
    const max = Math.max(...x);
    let sum = 0;
    for (let o = 0; o < x.length; o++) { x[o] = Math.exp(x[o] - max); sum += x[o]; }
    for (let o = 0; o < x.length; o++) x[o] /= sum;
    return x;
  }

  return { decodeBase64, loadModel, predict };
})();
