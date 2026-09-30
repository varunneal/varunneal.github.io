import type { FractalKernel, RenderRequest } from "./fractal"

// Keep this function self-contained: Quartz embeds it in a Blob worker, so no
// separate asset URL (or changes to the static site's build pipeline) is needed.
export function startFractalWorker(kernel: FractalKernel) {
  const scope = self as unknown as {
    onmessage: (event: MessageEvent<RenderRequest | null>) => void
    postMessage: (message: unknown, transfer: Transferable[]) => void
  }
  let generation = 0
  scope.onmessage = async ({ data }) => {
    const current = ++generation
    if (!data) return
    const frame = kernel.createFrame(data)
    let row = 0
    while (row < data.height) {
      const deadline = performance.now() + 8
      do {
        frame.renderRow(row++)
      } while (row < data.height && performance.now() < deadline)
      // Yield so a newer view can cancel this one, even during a deep zoom.
      await new Promise((resolve) => setTimeout(resolve, 0))
      if (current !== generation) return
    }
    const pixels = frame.finish()
    scope.postMessage({ id: data.id, width: data.width, height: data.height, pixels }, [
      pixels.buffer,
    ])
  }
}
