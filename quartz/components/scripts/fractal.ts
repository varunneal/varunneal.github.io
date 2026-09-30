export type DoubleDouble = [number, number]
export type FractalShape = "MANDELBROT" | "JULIA" | "SHIP"

export interface RenderRequest {
  id: number
  width: number
  height: number
  centerX: DoubleDouble
  centerY: DoubleDouble
  scale: number
  maxIter: number
  shape: FractalShape
  /** Width in render pixels; supersampling preserves the visible stroke weight. */
  edgeWidth?: number
}

export type FractalKernel = ReturnType<typeof createFractalKernel>

// Everything the worker needs lives inside this factory. It is serialized after
// bundling; do not capture module variables or import runtime dependencies here.
export function createFractalKernel() {
  function add(a: DoubleDouble, b: DoubleDouble): DoubleDouble {
    const s = a[0] + b[0]
    const v = s - a[0]
    const e = a[0] - (s - v) + (b[0] - v) + a[1] + b[1]
    const hi = s + e
    return [hi, e - (hi - s)]
  }

  function multiply(a: DoubleDouble, b: DoubleDouble): DoubleDouble {
    const product = a[0] * b[0]
    const ca = 134217729 * a[0]
    const cb = 134217729 * b[0]
    const ah = ca - (ca - a[0])
    const bh = cb - (cb - b[0])
    const al = a[0] - ah
    const bl = b[0] - bh
    const error =
      ah * bh - product + ah * bl + al * bh + al * bl + a[0] * b[1] + a[1] * b[0] + a[1] * b[1]
    const hi = product + error
    return [hi, error - (hi - product)]
  }

  function subtract(a: DoubleDouble, b: DoubleDouble): DoubleDouble {
    return add(a, [-b[0], -b[1]])
  }

  function inside(x: number, y: number) {
    const yy = y * y
    const q = (x - 0.25) * (x - 0.25) + yy
    // A conservative margin avoids falsely accepting points on the boundary.
    return q * (q + x - 0.25) < 0.25 * yy - 1e-15 || (x + 1) * (x + 1) + yy < 0.0625 - 1e-15
  }

  function escape(x0: number, y0: number, maxIter: number, shape: FractalShape) {
    if (shape === "MANDELBROT" && inside(x0, y0)) return maxIter
    let x = x0,
      y = y0,
      xx = x * x,
      yy = y * y,
      i = 0
    const cx = shape === "JULIA" ? -0.7 : x0
    const cy = shape === "JULIA" ? 0.27015 : y0
    const ship = shape === "SHIP"
    const radius = ship ? 3 : 4
    let savedX = x,
      savedY = y,
      period = 8,
      nextCheck = 8
    while (xx + yy <= radius && i < maxIter) {
      const end = Math.min(i + 32, maxIter)
      // Hoist formula dispatch and periodicity checks out of the hot loop.
      if (ship) {
        while (xx + yy <= radius && i < end) {
          y = Math.abs(2 * x * y) + cy
          x = xx - yy + cx
          xx = x * x
          yy = y * y
          i++
        }
      } else {
        while (xx + yy <= 4 && i < end) {
          y = 2 * x * y + cy
          x = xx - yy + cx
          xx = x * x
          yy = y * y
          i++
        }
      }
      // Exact repetition only: an epsilon test erases detail near the boundary.
      if (x === savedX && y === savedY) return maxIter
      if (i >= nextCheck) {
        savedX = x
        savedY = y
        period *= 2
        nextCheck += period
      }
    }
    return i
  }

  function reference(cx: DoubleDouble, cy: DoubleDouble, maxIter: number) {
    const real = new Float64Array(maxIter + 2)
    const imag = new Float64Array(maxIter + 2)
    const realLow = new Float64Array(maxIter + 2)
    const imagLow = new Float64Array(maxIter + 2)
    let x: DoubleDouble = [0, 0],
      y: DoubleDouble = [0, 0]
    let length = 0
    for (let i = 1; i < real.length; i++) {
      const nextX = add(subtract(multiply(x, x), multiply(y, y)), cx)
      const xy = multiply(x, y)
      y = add(add(xy, xy), cy)
      x = nextX
      real[i] = x[0] + x[1]
      imag[i] = y[0] + y[1]
      realLow[i] = x[1]
      imagLow[i] = y[1]
      length = i
      if (real[i] * real[i] + imag[i] * imag[i] > 4) break
    }
    return { real, imag, realLow, imagLow, length }
  }

  // delta(z[n+1]) = 2 Z[n] delta(z[n]) + delta(z[n])² + delta(c).
  // Rebase at the critical point, or when the reference runs out, to avoid
  // cancellation glitches: https://mathr.co.uk/blog/2022-02-21_deep_zoom_theory_and_practice_again.html
  function perturb(dx: number, dy: number, orbit: ReturnType<typeof reference>, maxIter: number) {
    let x = dx,
      y = dy,
      j = 1,
      i = 0
    let zx = orbit.real[j] + x,
      zy = orbit.imag[j] + y
    while (i < maxIter) {
      const magnitude = zx * zx + zy * zy
      if (Math.abs(magnitude - 4) < 1e-14) {
        // A rounded radius of exactly 2 can hide escape at extreme zooms
        // (especially near c = -2). Resolve only these rare ties at full precision.
        const preciseX = add([orbit.real[j], orbit.realLow[j]], [x, 0])
        const preciseY = add([orbit.imag[j], orbit.imagLow[j]], [y, 0])
        const radius = subtract(
          add(multiply(preciseX, preciseX), multiply(preciseY, preciseY)),
          [4, 0],
        )
        if (radius[0] > 0 || (radius[0] === 0 && radius[1] > 0)) break
      } else if (magnitude > 4) break
      if (zx * zx + zy * zy < x * x + y * y || j >= orbit.length) {
        x = zx
        y = zy
        j = 0
      }
      const nextX = 2 * (orbit.real[j] * x - orbit.imag[j] * y) + x * x - y * y + dx
      y = 2 * (orbit.real[j] * y + orbit.imag[j] * x + x * y) + dy
      x = nextX
      j++
      i++
      zx = orbit.real[j] + x
      zy = orbit.imag[j] + y
    }
    return i
  }

  function createFrame(view: RenderRequest) {
    const { width, height, centerX, centerY, scale, maxIter, shape, edgeWidth = 1 } = view
    const counts = new Uint32Array(width * height)
    const step = scale / height
    const orbit =
      shape === "MANDELBROT" && scale < 1e-5 ? reference(centerX, centerY, maxIter) : null
    const left = new Int32Array(height)
    const right = new Int32Array(height)
    for (let row = 0; row < height; row++) {
      const y = (row - height / 2) / (height / 2)
      const radius = (width / 2) * Math.sqrt(Math.max(0, 1 - y * y))
      left[row] = Math.max(0, Math.ceil(width / 2 - radius))
      right[row] = Math.min(width - 1 - edgeWidth, Math.floor(width / 2 + radius))
    }
    function renderRow(py: number) {
      // Include the right and lower neighbours needed by the edge pass.
      const start = Math.min(left[py], py >= edgeWidth ? left[py - edgeWidth] : width)
      const end = Math.min(
        width - 1,
        Math.max(right[py] + edgeWidth, py >= edgeWidth ? right[py - edgeWidth] : 0),
      )
      const dy = (py - height / 2) * step
      for (let px = start; px <= end; px++) {
        const dx = (px - width / 2) * step
        counts[py * width + px] = orbit
          ? perturb(dx, dy, orbit, maxIter)
          : escape(centerX[0] + (centerX[1] + dx), centerY[0] + (centerY[1] + dy), maxIter, shape)
      }
    }
    function finish() {
      // Transfer an alpha mask. Theme changes can recolor it without recomputing.
      const pixels = new Uint8ClampedArray(width * height * 4)
      for (let py = 0; py < height - edgeWidth; py++) {
        for (let px = left[py]; px <= right[py]; px++) {
          const idx = py * width + px
          if (
            counts[idx] !== counts[idx + edgeWidth] ||
            counts[idx] !== counts[idx + width * edgeWidth]
          ) {
            pixels[idx * 4 + 3] = 255
          }
        }
      }
      return pixels
    }
    return { renderRow, finish }
  }

  return { add, multiply, subtract, escape, reference, perturb, createFrame }
}
