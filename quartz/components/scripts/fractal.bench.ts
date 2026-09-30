// Run with: npx tsx quartz/components/scripts/fractal.bench.ts
import { createFractalKernel, type RenderRequest } from "./fractal"

const kernel = createFractalKernel()
const width = 289,
  height = 289
const views: [string, number, number, number, number][] = [
  ["Overview", 0, 0, 5, 100],
  ["Interior", 0, 0, 0.5, 100],
  ["Boundary", -0.743643887037151, 0.13182590420533, 0.0001, 600],
]

// Original per-pixel algorithm, including the edge pass and circular mask.
function original(view: RenderRequest) {
  const { width: w, height: h, maxIter, scale } = view
  const counts = new Uint16Array(w * h)
  const pixels = new Uint8ClampedArray(w * h * 4)
  const iterate = (x: number, y: number, cx: number, cy: number) => [
    x * x - y * y + cx,
    2 * x * y + cy,
  ]
  const stop = (x: number, y: number) => x * x + y * y > 4
  for (let py = 0; py < h; py++) {
    const cy = view.centerY[0] + (py - h / 2) * (scale / h)
    for (let px = 0; px < w; px++) {
      const cx = view.centerX[0] + (px - w / 2) * (scale / h)
      let x = cx,
        y = cy,
        i = 0
      while (!stop(x, y) && i < maxIter) {
        ;[x, y] = iterate(x, y, cx, cy)
        i++
      }
      counts[py * w + px] = i
    }
  }
  for (let py = 0; py < h - 1; py++) {
    for (let px = 0; px < w - 1; px++) {
      const idx = py * w + px
      if (counts[idx] !== counts[idx + 1] || counts[idx] !== counts[idx + w]) {
        pixels[idx * 4 + 3] = 255
        if (((px - w / 2) * h) ** 2 + ((py - h / 2) * w) ** 2 > ((w * h) / 2) ** 2)
          pixels[idx * 4 + 3] = 0
      }
    }
  }
  return pixels
}
function optimized(view: RenderRequest) {
  const frame = kernel.createFrame({
    ...view,
    width: view.width * 2,
    height: view.height * 2,
    edgeWidth: 2,
  })
  for (let row = 0; row < view.height * 2; row++) frame.renderRow(row)
  return frame.finish()
}
function median(fn: () => unknown) {
  for (let i = 0; i < 3; i++) fn()
  const timings: number[] = []
  for (let i = 0; i < 9; i++) {
    const start = performance.now()
    fn()
    timings.push(performance.now() - start)
  }
  return timings.sort((a, b) => a - b)[4]
}
console.table(
  views.map(([name, x, y, scale, maxIter]) => {
    const view: RenderRequest = {
      id: 1,
      width,
      height,
      centerX: [x, 0],
      centerY: [y, 0],
      scale,
      maxIter,
      shape: "MANDELBROT",
    }
    const before = median(() => original(view))
    const after = median(() => optimized(view))
    return {
      view: name,
      "original ms": before.toFixed(2),
      "new supersampled ms": after.toFixed(2),
      "full speedup": (before / after).toFixed(1) + "×",
    }
  }),
)
