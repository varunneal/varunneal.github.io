import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import style from "./styles/articleImage.scss"
// import { i18n } from "../i18n"

interface ArticleImageOptions {
    // Configuration options
    defaultImage?: string
    maxWidth?: string
    maxHeight?: string
    // The frontmatter fields to check for image paths
    frontmatterField?: string
    darkImageField?: string
    lightImageField?: string
}

const defaultOptions: ArticleImageOptions = {
    defaultImage: "",
    maxWidth: "100%",
    maxHeight: "150px",
    frontmatterField: "image",
    darkImageField: "darkImage",
    lightImageField: "lightImage",
}

export default ((opts?: Partial<ArticleImageOptions>) => {
    const options = { ...defaultOptions, ...opts }

    const ArticleImage: QuartzComponent = ({
                                               fileData,
                                               displayClass,
                                               cfg,
                                           }: QuartzComponentProps) => {
        if (!fileData) return null

        // Get image paths from frontmatter
        let imagePath = null
        let darkImagePath = null
        let lightImagePath = null
        let imageAlt = null

        if (fileData.frontmatter) {
            // Process standard image
            if (options.frontmatterField && fileData.frontmatter[options.frontmatterField]) {
                // It could be just a string with the path
                if (typeof fileData.frontmatter[options.frontmatterField] === "string") {
                    imagePath = fileData.frontmatter[options.frontmatterField]
                }
                // Or it could be an object with path and alt
                else if (typeof fileData.frontmatter[options.frontmatterField] === "object") {
                    const imgData = fileData.frontmatter[options.frontmatterField]
                    imagePath = imgData.path || imgData.src || imgData.url
                    imageAlt = imgData.alt || imgData.description || null
                }
            }

            // Process dark theme image
            if (options.darkImageField && fileData.frontmatter[options.darkImageField]) {
                if (typeof fileData.frontmatter[options.darkImageField] === "string") {
                    darkImagePath = fileData.frontmatter[options.darkImageField]
                } else if (typeof fileData.frontmatter[options.darkImageField] === "object") {
                    const imgData = fileData.frontmatter[options.darkImageField]
                    darkImagePath = imgData.path || imgData.src || imgData.url
                }
            }

            // Process light theme image
            if (options.lightImageField && fileData.frontmatter[options.lightImageField]) {
                if (typeof fileData.frontmatter[options.lightImageField] === "string") {
                    lightImagePath = fileData.frontmatter[options.lightImageField]
                } else if (typeof fileData.frontmatter[options.lightImageField] === "object") {
                    const imgData = fileData.frontmatter[options.lightImageField]
                    lightImagePath = imgData.path || imgData.src || imgData.url
                }
            }

            // If no specific alt text was provided, check for a dedicated alt field
            if (!imageAlt && fileData.frontmatter.imageAlt) {
                imageAlt = fileData.frontmatter.imageAlt
            }
        }

        // Look for the image in content if using a special syntax: ![[image:path/to/image.jpg]]
        if (!imagePath && fileData.content) {
            const regex = /!\[\[image:(.*?)\]\]/i
            const match = fileData.content.match(regex)
            if (match && match[1]) {
                imagePath = match[1].trim()
            }
        }

        // Check for raw image (no theming)
        const rawImagePath = fileData.frontmatter?.imageRaw as string | undefined
        const isRaw = !!rawImagePath
        if (isRaw) {
            imagePath = rawImagePath
        }

        // If we still don't have an image, use the default if provided
        if (!imagePath && options.defaultImage) {
            imagePath = options.defaultImage
        }

        // If there's no image to display, return null
        if (!imagePath && !darkImagePath && !lightImagePath) {
            return null
        }

        // Determine which classes to apply based on theme-specific images
        const hasThemeImages = darkImagePath && lightImagePath
        const imageStyle = fileData.frontmatter?.imageStyle as string | undefined

        return (
            <div class={`${displayClass} article-image`}>
                {hasThemeImages ? (
                    <>
                        <img
                            src={lightImagePath}
                            alt={imageAlt}
                            class={`light-theme-image ${imageStyle ?? ""}`}
                        />
                        <img
                            src={darkImagePath}
                            alt={imageAlt}
                            class={`dark-theme-image ${imageStyle ?? ""}`}
                        />
                    </>
                ) : (
                    <img
                        src={imagePath}
                        alt={imageAlt}
                        class={`${isRaw ? "" : "auto-themed-image"} ${imageStyle ?? ""}`}
                    />
                )}
            </div>
        )
    }

    // Attach CSS
    ArticleImage.css = style
    return ArticleImage
}) satisfies QuartzComponentConstructor
