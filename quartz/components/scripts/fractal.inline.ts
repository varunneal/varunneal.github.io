type IterFn = (
    x: number,
    y: number,
    x0: number,
    y0: number,
    iterations: number
) => [number, number];

type StopCondition = (
    x: number,
    y: number,
    x0: number,
    y0: number,
    iterations: number
) => boolean;

interface FractalDef {
    iterFn: IterFn;
    stopCondition: StopCondition;
}


export const FRACTALS: Record<string, FractalDef> = {
    MANDELBROT: {
        iterFn: (x, y, x0, y0 /* iterations */) => [
            x * x - y * y + x0,
            2 * x * y + y0,
        ],
        stopCondition: (x, y /* x0, y0, iterations */) => x * x + y * y > 4,
    },

    // Example extras — comment out or tweak as needed
    JULIA: {
        iterFn: (x, y, _x0, _y0, _iter) => {
            const cx = -0.7;
            const cy = 0.27015;
            return [x * x - y * y + cx, 2 * x * y + cy];
        },
        stopCondition: (x, y) => x * x + y * y > 4,
    },

    SHIP: {
        iterFn: (x, y, x0, y0) => [
            x * x - y * y + x0,
            Math.abs(2 * x * y) + y0,
        ],
        stopCondition: (x, y) => x * x + y * y > 3,
    },
};

(() => {
    let cleanup: (() => void) | null = null
    function mount() {
        const cvs = document.getElementById("fractal-canvas") as HTMLCanvasElement
        if (!cvs) return //console.error("[Fractal] canvas not found")

        const ctx = cvs.getContext("2d")!
        const W = cvs.width
        const H = cvs.height
        const maxIter = Number(cvs.dataset.maxIter) || 100

        const iters = new Uint16Array(W * H)
        const img = ctx.createImageData(W, H)
        const rgba = new Uint8ClampedArray(img.data.buffer)

        /* view ------------------------------------------------------- */
        let centerX = 0
        let centerY = 0
        let scale = Number(cvs.dataset.initScale) || 5

        const minScale = Number(cvs.dataset.minScale) || 1e-6
        const maxScale = Number(cvs.dataset.maxScale) || 1e3
        const zoomFactor = Number(cvs.dataset.zoomFactor) || 0.98

        const fractalShape = String(cvs.dataset.fractalShape) || "Mandelbrot"

        let dirty = true

        /* hover-zoom state ------------------------------------------ */
        let hoverX = 0, hoverY = 0
        let zoomActive = false
        let hoverTimer: number | null = null
        const HOVER_DELAY = 150      // ms before zoom kicks in
        const MOVE_TOL = 4        // px wiggle room

        /* math ------------------------------------------------------- */

        const {iterFn, stopCondition} = FRACTALS[fractalShape.toUpperCase()]

        function compute() {
            const s = scale / H
            for (let py = 0; py < H; ++py) {
                const y0 = centerY + (py - H / 2) * s
                for (let px = 0; px < W; ++px) {
                    const x0 = centerX + (px - W / 2) * s
                    let x = x0, y = y0, i = 0

                    // maxIter is enforced, not necessary in stopCondition
                    while (!stopCondition(x, y, x0, y0, i) && i < maxIter) {
                        [x, y] = iterFn(x, y, x0, y0, i)
                        ++i
                    }
                    iters[py * W + px] = i
                }
            }

            let hex = getComputedStyle(cvs)
                .getPropertyValue("--edge-color")
                .trim()
                .replace("#", "")
            const r = parseInt(hex.substring(0, 2), 16);
            const g = parseInt(hex.substring(2, 4), 16);
            const b = parseInt(hex.substring(4, 6), 16);


            for (let py = 0; py < H - 1; ++py) {
                for (let px = 0; px < W - 1; ++px) {
                    const idx = py * W + px
                    const edge =
                        iters[idx] !== iters[idx + 1] ||
                        iters[idx] !== iters[idx + W]
                    const off = idx << 2
                    if (edge) {
                        rgba[off] = r
                        rgba[off + 1] = g
                        rgba[off + 2] = b
                        rgba[off + 3] = 255
                        if (((px - W / 2) * H) ** 2 + ((py - H / 2) * W) ** 2 > (W * H / 2) ** 2) {
                            rgba[off + 3] = 0;
                        }
                    } else {
                        rgba[off + 3] = 0
                    }
                }
            }
        }

        /* RAF control ------------------------------------------------ */
        /* (render loop + "request animation frame" control ) */
        let raf = 0
        const run = () => {
            if (zoomActive && scale > minScale && scale < maxScale) {
                // move center toward hover point, then shrink scale
                const fx = (hoverX - W / 2) / H
                const fy = (hoverY - H / 2) / H
                centerX += fx * scale * (1 - zoomFactor)  // 1/x ≈ 1-x for x ≈ 1
                centerY += fy * scale * (1 - zoomFactor)
                scale *= zoomFactor
                dirty = true
            } else {
                zoomActive = false
            }

            if (dirty) {
                compute()
                ctx.putImageData(img, 0, 0)
                dirty = false
            }
            // keep the loop only if there’s work to do
            if (zoomActive || dirty) raf = requestAnimationFrame(run)
            else raf = 0                         // loop goes to sleep
        }

        const wake = () => {
            if (!raf) raf = requestAnimationFrame(run)
        }
        const sleep = () => {
            if (raf) cancelAnimationFrame(raf);
            raf = 0
        }

        /* helpers ---------------------------------------------------- */
        const startHoverTimer = (x: number, y: number) => {
            clearTimeout(hoverTimer!)
            hoverTimer = window.setTimeout(() => {
                hoverX = x;
                hoverY = y
                zoomActive = true
            }, HOVER_DELAY)
        }

        const cancelHover = () => {
            clearTimeout(hoverTimer!)
            zoomActive = false
        }

        /* events ----------------------------------------------------- */
        cvs.addEventListener("mouseenter", e => {
            const {left, top} = cvs.getBoundingClientRect()
            startHoverTimer(e.clientX - left, e.clientY - top)
            wake()
        })

        cvs.addEventListener("mousemove", e => {
            const {left, top} = cvs.getBoundingClientRect()
            const x = e.clientX - left
            const y = e.clientY - top

            // if we're already zooming, update hover target smoothly
            if (zoomActive) {
                hoverX = x
                hoverY = y
            } else {
                const dx = x - hoverX
                const dy = y - hoverY
                if (dx * dx + dy * dy > MOVE_TOL * MOVE_TOL) {
                    hoverX = x;
                    hoverY = y
                    cancelHover()
                    startHoverTimer(x, y)
                }
            }
            wake()
        })

        cvs.addEventListener("mouseleave", () => {
            cancelHover()
            sleep()
        })

        // manual scroll zoom / pan still work
        cvs.addEventListener("wheel", e => {
            e.preventDefault()
            if (scale > minScale && e.deltaY < 0) {
                scale *= 0.9
            } else if (scale < maxScale && e.deltaY > 0) {
                scale *= 1.1
            }
            dirty = true

            cancelHover()
            wake()
        }, {passive: false})

        cvs.addEventListener("mousedown", () => {
            cancelHover()
            wake()
        })
        cvs.addEventListener("mousemove", e => {
            if (e.buttons === 1) {
                centerX -= (e.movementX / W) * scale
                centerY -= (e.movementY / H) * scale
                dirty = true
            }
            wake()
        })



        compute()
        ctx.putImageData(img, 0, 0)

        cleanup = () => {
            cancelAnimationFrame(raf)
            cvs.replaceWith(cvs.cloneNode(true))  // drops listeners
            cleanup = null
        }
    }
    /* go --------------------------------------------------------- */

    document.addEventListener("nav", () => {
        cleanup?.()
        mount()
    })

    document.addEventListener("themechange", () => {
        mount()
    })
    mount()


})()
