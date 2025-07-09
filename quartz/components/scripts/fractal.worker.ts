/// <reference lib="webworker" />

/** message coming from main thread */
interface Payload {
    width: number
    height: number
    center: { x: number; y: number }
    scale: number
    maxIter: number
}

self.onmessage = ({ data }: MessageEvent<Payload>) => {
    const { width, height, center, scale, maxIter } = data
    const buf = new Uint32Array(width * height)

    for (let py = 0; py < height; py++) {
        for (let px = 0; px < width; px++) {
            // map pixel → complex plane
            let x0 =
                center.x + ((px - width / 2) * scale) / height
            let y0 =
                center.y + ((py - height / 2) * scale) / height
            let x = x0,
                y = y0,
                iter = 0

            while (x * x + y * y <= 4 && iter < maxIter) {
                const xt = x * x - y * y + x0
                y = 2 * x * y + y0
                x = xt
                iter++
            }

            // edge mask: look for pixels that *didn’t* escape
            buf[py * width + px] = iter === maxIter ? 0 : 0xff_00_00_00
        }
    }

    /* second pass for edge detection (neighbour compare) */
    for (let py = 0; py < height - 1; py++) {
        for (let px = 0; px < width - 1; px++) {
            const idx = py * width + px
            if (
                buf[idx] !== buf[idx + 1] ||
                buf[idx] !== buf[idx + width]
            ) {
                buf[idx] = 0xff_00_00_00 // opaque black
            } else {
                buf[idx] = 0 // transparent
            }
        }
    }

    /* ship the raw RGBA back */
    // NB: buf is already in ABGR byte order for ImageData
    self.postMessage(buf.buffer, [buf.buffer])
}
