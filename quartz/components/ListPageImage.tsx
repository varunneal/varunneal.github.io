import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"

// Simple image component for list/folder pages - displays at bottom with no theme switching
export default (() => {
  const ListPageImage: QuartzComponent = ({ fileData }: QuartzComponentProps) => {
    if (!fileData?.frontmatter) return null

    const imagePath = fileData.frontmatter.image
    if (!imagePath || typeof imagePath !== "string") return null

    return (
      <div class="list-page-image">
        <img src={imagePath} alt="" />
      </div>
    )
  }

  ListPageImage.css = `
    .list-page-image {
      margin-top: 2rem;
      width: 100%;
    }
    .list-page-image img {
      width: 100%;
      opacity: 0.85;
    }
  `

  return ListPageImage
}) satisfies QuartzComponentConstructor
