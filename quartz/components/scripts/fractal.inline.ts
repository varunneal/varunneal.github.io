import {
  createFractalKernel,
  type DoubleDouble,
  type FractalShape,
  type RenderRequest,
} from "./fractal"
import { startFractalWorker } from "./fractal.worker"
;(() => {
  const kernel = createFractalKernel()
  let cleanup: (() => void) | undefined

  function mount() {
    cleanup?.()
    cleanup = undefined
    const canvas = document.getElementById("fractal-canvas") as HTMLCanvasElement | null
    const ctx = canvas?.getContext("2d")
    if (!canvas || !ctx) return
    const cvs = canvas
    const context = ctx
    const width = cvs.width,
      height = cvs.height
    const supersampling = 2
    const initialScale = Number(cvs.dataset.initScale) || 5
    const shapeName = cvs.dataset.fractalShape?.toUpperCase()
    const shape: FractalShape =
      shapeName === "JULIA" || shapeName === "SHIP" ? shapeName : "MANDELBROT"
    // Mandelbrot uses ~106-bit coordinates. Other formulas retain the ordinary
    // double renderer, so stop before adjacent pixels collapse to one number.
    const minScale = Math.max(
      Number(cvs.dataset.minScale) || 1e-27,
      shape === "MANDELBROT" ? 1e-27 : 1e-12,
    )
    const maxScale = Math.max(minScale, Number(cvs.dataset.maxScale) || 1e3)
    const baseIter = Math.max(1, Number(cvs.dataset.maxIter) || 100)
    const iterLimit = Math.max(baseIter, Number(cvs.dataset.maxIterLimit) || 8192)
    const zoomFactor = Math.max(0.5, Math.min(1, Number(cvs.dataset.zoomFactor) || 0.98))
    let centerX: DoubleDouble = [0, 0],
      centerY: DoubleDouble = [0, 0]
    let scale = Math.max(minScale, Math.min(maxScale, initialScale))
    let hoverX = width / 2,
      hoverY = height / 2
    let zoomActive = false,
      dragging = false,
      dirty = true,
      busy = false
    let disposed = false,
      inViewport = true
    let raf = 0,
      requestId = 0,
      lastZoom = 0,
      lastRender = 0
    let hoverTimer: number | undefined
    let dragX = 0,
      dragY = 0
    let mask: ImageData | undefined
    let worker: Worker | undefined
    let workerURL: string | undefined
    let pending: RenderRequest | undefined
    const abort = new AbortController()
    const signal = abort.signal
    const buffer = document.createElement("canvas")
    const bufferContext = buffer.getContext("2d")!

    const visible = () => inViewport && !document.hidden && cvs.getClientRects().length > 0
    const wake = () => {
      if (!disposed && !raf && visible()) raf = requestAnimationFrame(run)
    }
    const cancelHover = () => {
      clearTimeout(hoverTimer)
      hoverTimer = undefined
      zoomActive = false
      lastZoom = 0
    }
    const position = (e: MouseEvent) => {
      const rect = cvs.getBoundingClientRect()
      return [
        ((e.clientX - rect.left) * width) / rect.width,
        ((e.clientY - rect.top) * height) / rect.height,
      ]
    }
    const zoom = (factor: number, x: number, y: number) => {
      const next = Math.max(minScale, Math.min(maxScale, scale * factor))
      const delta = scale - next
      centerX = kernel.add(centerX, [((x - width / 2) / height) * delta, 0])
      centerY = kernel.add(centerY, [((y - height / 2) / height) * delta, 0])
      scale = next
      dirty = true
    }
    const startHover = (x: number, y: number) => {
      cancelHover()
      hoverX = x
      hoverY = y
      if (zoomFactor === 1) return
      hoverTimer = window.setTimeout(() => {
        hoverTimer = undefined
        zoomActive = true
        lastZoom = performance.now()
        wake()
      }, 150)
    }

    function paint() {
      if (!mask) return
      buffer.width = mask.width
      buffer.height = mask.height
      bufferContext.putImageData(mask, 0, 0)
      bufferContext.globalCompositeOperation = "source-in"
      bufferContext.fillStyle =
        getComputedStyle(cvs).getPropertyValue("--edge-color").trim() || "#56473a"
      bufferContext.fillRect(0, 0, buffer.width, buffer.height)
      bufferContext.globalCompositeOperation = "source-over"
      context.clearRect(0, 0, width, height)
      context.imageSmoothingEnabled = true
      context.imageSmoothingQuality = "high"
      context.drawImage(buffer, 0, 0, width, height)
    }

    function receive(data: {
      id: number
      width: number
      height: number
      pixels: Uint8ClampedArray<ArrayBuffer>
    }) {
      if (disposed || data.id !== requestId) return
      busy = false
      pending = undefined
      mask = new ImageData(data.pixels, data.width, data.height)
      paint()
      if (zoomActive || dirty) wake()
    }

    function fallback(view: RenderRequest) {
      const frame = kernel.createFrame(view)
      let row = 0
      const chunk = () => {
        if (disposed || view.id !== requestId || !visible()) return
        const deadline = performance.now() + 6
        do {
          frame.renderRow(row++)
        } while (row < view.height && performance.now() < deadline)
        if (row < view.height) window.setTimeout(chunk, 0)
        else receive({ ...view, pixels: frame.finish() })
      }
      window.setTimeout(chunk, 0)
    }

    function stopWorker() {
      worker?.terminate()
      worker = undefined
      if (workerURL) URL.revokeObjectURL(workerURL)
      workerURL = undefined
    }

    try {
      workerURL = URL.createObjectURL(
        new Blob([`(${startFractalWorker.toString()})((${createFractalKernel.toString()})())`], {
          type: "text/javascript",
        }),
      )
      worker = new Worker(workerURL)
      worker.onmessage = ({ data }) => receive(data)
      worker.onerror = (event) => {
        event.preventDefault()
        stopWorker()
        if (pending) fallback(pending)
      }
    } catch {
      stopWorker()
    }

    function run(now: number) {
      raf = 0
      if (!visible()) return
      if (zoomActive && !busy) {
        if (scale > minScale) {
          const elapsed = Math.min(50, now - (lastZoom || now - 1000 / 60))
          zoom(zoomFactor ** (elapsed / (1000 / 60)), hoverX, hoverY)
          lastZoom = now
        } else {
          cancelHover()
        }
      }
      if (dirty) {
        // Cap rendering at 30 fps; worker completion paces sustained hover zoom.
        if (now - lastRender < 1000 / 30) {
          wake()
          return
        }
        lastRender = now
        pending = {
          id: ++requestId,
          width: width * supersampling,
          height: height * supersampling,
          edgeWidth: supersampling,
          centerX,
          centerY,
          scale,
          shape,
          maxIter: Math.min(
            iterLimit,
            baseIter + Math.ceil(Math.max(0, Math.log2(initialScale / scale))) * 32,
          ),
        }
        dirty = false
        busy = true
        if (worker) worker.postMessage(pending)
        else fallback(pending)
      }
    }

    cvs.addEventListener(
      "pointerenter",
      (e) => {
        if (e.pointerType === "mouse" && !dragging) {
          const [x, y] = position(e)
          startHover(x, y)
        }
      },
      { signal },
    )
    cvs.addEventListener(
      "pointermove",
      (e) => {
        const [x, y] = position(e)
        if (dragging) {
          centerX = kernel.add(centerX, [(-(x - dragX) / height) * scale, 0])
          centerY = kernel.add(centerY, [(-(y - dragY) / height) * scale, 0])
          dragX = x
          dragY = y
          dirty = true
          wake()
        } else if (e.pointerType === "mouse") {
          if (zoomActive) {
            hoverX = x
            hoverY = y
          } else if ((x - hoverX) ** 2 + (y - hoverY) ** 2 > 16) startHover(x, y)
        }
      },
      { signal },
    )
    cvs.addEventListener("pointerleave", cancelHover, { signal })
    cvs.addEventListener(
      "pointerdown",
      (e) => {
        if (e.button !== 0) return
        cancelHover()
        dragging = true
        ;[dragX, dragY] = position(e)
        cvs.setPointerCapture(e.pointerId)
      },
      { signal },
    )
    const endDrag = () => {
      dragging = false
    }
    cvs.addEventListener("pointerup", endDrag, { signal })
    cvs.addEventListener("pointercancel", endDrag, { signal })
    cvs.addEventListener("lostpointercapture", endDrag, { signal })
    cvs.addEventListener(
      "wheel",
      (e) => {
        e.preventDefault()
        cancelHover()
        const [x, y] = position(e)
        const delta = e.deltaY * (e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? height : 1)
        zoom(Math.exp(Math.max(-1, Math.min(1, delta * 0.002))), x, y)
        wake()
      },
      { passive: false, signal },
    )
    const reset = () => {
      cancelHover()
      centerX = [0, 0]
      centerY = [0, 0]
      scale = Math.max(minScale, Math.min(maxScale, initialScale))
      dirty = true
      wake()
    }
    cvs.addEventListener("dblclick", reset, { signal })
    cvs.addEventListener(
      "keydown",
      (e) => {
        if (e.key === "Home" || e.key === "Escape") {
          e.preventDefault()
          reset()
        } else if (["+", "=", "-"].includes(e.key)) {
          e.preventDefault()
          cancelHover()
          zoom(e.key === "-" ? 1.25 : 0.8, width / 2, height / 2)
          wake()
        }
      },
      { signal },
    )
    document.addEventListener("themechange", paint, { signal })
    const visibilityChanged = () => {
      if (!visible()) {
        cancelHover()
        cancelAnimationFrame(raf)
        raf = 0
        requestId++
        worker?.postMessage(null)
        pending = undefined
        busy = false
      } else {
        dirty = true
        wake()
      }
    }
    document.addEventListener("visibilitychange", visibilityChanged, { signal })
    const observer = new IntersectionObserver(([entry]) => {
      inViewport = entry.isIntersecting
      visibilityChanged()
    })
    observer.observe(cvs)
    wake()
    cleanup = () => {
      disposed = true
      cancelHover()
      cancelAnimationFrame(raf)
      abort.abort()
      observer.disconnect()
      stopWorker()
    }
  }

  document.addEventListener("nav", mount)
  mount()
})()
