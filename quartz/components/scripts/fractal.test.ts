import assert from "node:assert/strict"
import { test } from "node:test"
import vm from "node:vm"
import { build } from "esbuild"
import {
  createFractalKernel,
  type DoubleDouble,
  type FractalShape,
  type RenderRequest,
} from "./fractal"

const kernel = createFractalKernel()

// The original renderer, kept independent of the optimized recurrence.
function baseline(x0: number, y0: number, maxIter: number, shape: FractalShape) {
  let x = x0,
    y = y0,
    i = 0
  while (x * x + y * y <= (shape === "SHIP" ? 3 : 4) && i < maxIter) {
    const nextX = x * x - y * y + (shape === "JULIA" ? -0.7 : x0)
    y = (shape === "SHIP" ? Math.abs(2 * x * y) : 2 * x * y) + (shape === "JULIA" ? 0.27015 : y0)
    x = nextX
    i++
  }
  return i
}

function render(view: RenderRequest) {
  const frame = kernel.createFrame(view)
  for (let row = 0; row < view.height; row++) frame.renderRow(row)
  return frame.finish()
}

const defaultView: RenderRequest = {
  id: 1,
  width: 81,
  height: 63,
  centerX: [-0.5, 0],
  centerY: [0, 0],
  scale: 5,
  maxIter: 200,
  shape: "MANDELBROT",
}

test("optimized escape counts and circular edges preserve all three formulas", () => {
  for (const shape of ["MANDELBROT", "JULIA", "SHIP"] as const) {
    const view = { ...defaultView, shape }
    const { width, height, scale, maxIter } = view
    const counts = new Uint32Array(width * height)
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const cx = view.centerX[0] + (x - width / 2) * (scale / height)
        const cy = (y - height / 2) * (scale / height)
        const expected = baseline(cx, cy, maxIter, shape)
        assert.equal(kernel.escape(cx, cy, maxIter, shape), expected)
        counts[y * width + x] = expected
      }
    }
    for (const edgeWidth of [1, 2]) {
      const pixels = render({ ...view, edgeWidth })
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const i = y * width + x
          const inside =
            ((x - width / 2) * height) ** 2 + ((y - height / 2) * width) ** 2 <=
            ((width * height) / 2) ** 2
          const edge =
            x < width - edgeWidth &&
            y < height - edgeWidth &&
            inside &&
            (counts[i] !== counts[i + edgeWidth] || counts[i] !== counts[i + width * edgeWidth])
          assert.equal(
            pixels[i * 4 + 3],
            edge ? 255 : 0,
            `${shape} edge width ${edgeWidth}, pixel ${x},${y}`,
          )
        }
      }
    }
  }
})

// Independent 192-bit fixed-point oracle; no double-double or perturbation math.
const bits = 192n
const unit = 1n << bits
const factor = 2 ** Number(bits)
const fixed = (n: number) => BigInt(Math.trunc(n * factor))
function decimal(value: string) {
  const [whole, fraction = ""] = value.replace("-", "").split(".")
  const result = (BigInt(whole + fraction) * unit) / 10n ** BigInt(fraction.length)
  return value.startsWith("-") ? -result : result
}
function precise(value: string): DoubleDouble {
  const hi = Number(value)
  return [hi, Number(decimal(value) - fixed(hi)) / factor]
}
function highPrecision(cx: bigint, cy: bigint, maxIter: number) {
  let x = cx,
    y = cy,
    i = 0
  while (x * x + y * y <= 4n * unit * unit && i < maxIter) {
    const nextX = ((x * x - y * y) >> bits) + cx
    y = ((2n * x * y) >> bits) + cy
    x = nextX
    i++
  }
  return i
}

test("coordinates retain movement far below double precision", () => {
  const initial: DoubleDouble = [-0.75, 0]
  let moved = initial
  for (let i = 0; i < 1000; i++) moved = kernel.add(moved, [1e-27, 0])
  assert.equal(moved[0], initial[0])
  assert.ok(Math.abs(moved[1] / 1e-24 - 1) < 1e-12)
  const restored = kernel.add(moved, [-1e-24, 0])
  assert.ok(Math.abs(restored[1]) < 1e-37)
})

test("deep perturbation agrees with 192-bit arithmetic and resolves distinct pixels", () => {
  // A real Misiurewicz point with structure at every tested depth.
  const cx = precise("-1.54368901269207636157085597180174798652520329765098")
  const cy: DoubleDouble = [0, 0]
  for (const scale of [1e-7, 1e-14, 1e-23, 1e-27]) {
    const orbit = kernel.reference(cx, cy, 1000)
    const distinct = new Set<number>()
    for (let py = -4; py <= 4; py++) {
      for (let px = -4; px <= 4; px++) {
        const dx = ((px + 0.25) * scale) / 9
        const dy = ((py + 0.25) * scale) / 9
        const expected = highPrecision(fixed(cx[0]) + fixed(cx[1]) + fixed(dx), fixed(dy), 1000)
        const actual = kernel.perturb(dx, dy, orbit, 1000)
        assert.equal(actual, expected, `scale=${scale}, pixel=${px},${py}`)
        distinct.add(actual)
      }
    }
    assert.ok(distinct.size > 3, `resolved detail at scale ${scale}`)
  }
})

test("perturbation rebases when a reference escapes before nearby pixels", () => {
  const cx: DoubleDouble = [-0.75, 0],
    cy: DoubleDouble = [0.1, 0]
  const orbit = kernel.reference(cx, cy, 1000)
  for (const [dx, dy] of [
    [0.01, -0.1],
    [0.001, -0.001],
    [-0.01, 0.01],
    [0, 0],
  ]) {
    const expected = highPrecision(fixed(cx[0]) + fixed(dx), fixed(cy[0]) + fixed(dy), 1000)
    assert.equal(kernel.perturb(dx, dy, orbit, 1000), expected)
  }
})

test("deep escape tests resolve offsets that round onto the bailout circle", () => {
  const orbit = kernel.reference([-2, 0], [0, 0], 1000)
  for (const dx of [-1e-27, 0, 1e-27]) {
    assert.equal(kernel.perturb(dx, 0, orbit, 1000), highPrecision(fixed(-2) + fixed(dx), 0n, 1000))
  }
})

test("minified Blob worker is self-contained, transfers frames and cancels stale work", async () => {
  const bundle = await build({
    stdin: {
      contents: `import { createFractalKernel } from './quartz/components/scripts/fractal';
        import { startFractalWorker } from './quartz/components/scripts/fractal.worker';
        globalThis.workerSource = '(' + startFractalWorker.toString() + ')((' + createFractalKernel.toString() + ')())';`,
      resolveDir: process.cwd(),
      loader: "ts",
    },
    bundle: true,
    minify: true,
    write: false,
    format: "iife",
  })
  const outer = { workerSource: "" }
  vm.runInNewContext(bundle.outputFiles[0].text, outer)
  const replies: { id: number; pixels: Uint8ClampedArray }[] = []
  const scope = {
    onmessage: async (_event: { data: RenderRequest | null }) => {},
    postMessage: (reply: (typeof replies)[number], transfers: ArrayBuffer[]) => {
      assert.equal(transfers[0], reply.pixels.buffer)
      replies.push(reply)
    },
  }
  vm.runInNewContext(outer.workerSource, { self: scope, performance, setTimeout })
  const stale = scope.onmessage({ data: { ...defaultView, id: 1, maxIter: 10000 } })
  const latest = scope.onmessage({ data: { ...defaultView, id: 2 } })
  await Promise.all([stale, latest])
  assert.deepEqual(
    replies.map((r) => r.id),
    [2],
  )
  assert.deepEqual(Array.from(replies[0].pixels), Array.from(render(defaultView)))
  const cancelled = scope.onmessage({ data: { ...defaultView, id: 3 } })
  await scope.onmessage({ data: null })
  await cancelled
  assert.equal(replies.length, 1)
})
