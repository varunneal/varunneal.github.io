import {
    QuartzComponent,
    QuartzComponentConstructor,
    QuartzComponentProps,
} from "./types"

// @ts-ignore
import script from "./scripts/fractal.inline" // after-load logic
import styles from "./styles/fractal.scss"    // optional CSS

/** public config for people who instantiate <Fractal opts={…}/> */
export interface Options {
    width?: number
    height?: number
    maxIter?: number

    initScale?: number
    minScale?: number
    maxScale?: number

    /* Between 0 and 1. zoomFactor = 1 is no Zoom. */
    zoomFactor?: number

    // iterFn?: (x: number, y: number, x0: number, y0: number, iterations: number) => [xn: number, yn: number]
    // stopCondition?: (x: number, y: number, x0: number, y0: number, iterations: number) => boolean
}

const defaultOpts: Required<Options> = {
    width: 289,
    height: 289,

    maxIter: 100,
    initScale: 5,
    minScale: 1e-9,
    maxScale: 1e3,

    zoomFactor: 0.98,
}

const makeFractal = (opts: Options): QuartzComponent => {
    const cfg = { ...defaultOpts, ...opts }

    const Fractal = (_props: QuartzComponentProps) => {
        const fractal = _props.fileData.frontmatter?.fractal as string | undefined
        if (!fractal) return null;

        return (
            <canvas
                id="fractal-canvas"
                width={cfg.width}
                height={cfg.height}

                data-max-iter={cfg.maxIter}
                data-init-scale={cfg.initScale}
                data-min-scale={cfg.minScale}
                data-max-scale={cfg.maxScale}
                data-zoom-factor={cfg.zoomFactor}

                data-fractal-shape={fractal}
            />
        )
    }

    /* attach assets so Quartz picks them up */
    Fractal.afterDOMLoaded = script
    Fractal.css = styles

    return Fractal
}

const Fractal = (
    userOpts: Options = {},
) => makeFractal(userOpts)

const typedFractal = Fractal satisfies QuartzComponentConstructor

export default typedFractal;

