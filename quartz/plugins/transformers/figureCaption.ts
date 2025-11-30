import { QuartzTransformerPlugin } from "../types"
import { Root } from "mdast"
import { visit } from "unist-util-visit"

export const FigureCaption: QuartzTransformerPlugin = () => ({
  name: "FigureCaption",
  markdownPlugins() {
    return [
      () => (tree: Root) => {
        visit(tree, "strong", (node, index, parent) => {
          if (!parent || index === undefined) return

          // Check if the strong node contains text starting with "Figure X:" or "Algorithm X:"
          const firstChild = node.children[0]
          if (firstChild?.type !== "text") return

          const text = firstChild.value
          const match = text.match(/^(Figure\s+\d+|Algorithm\s+\d+)(:.*)?$/i)
          if (!match) return

          const label = match[1] // "Figure 1" or "Algorithm 1"
          const rest = match[2] || "" // ": description..." or ""

          // Replace the strong node with HTML containing smallcaps
          const remainingChildren = node.children.slice(1)
          const remainingText = remainingChildren
            .map((child: any) => child.value || "")
            .join("")

          const htmlNode = {
            type: "html" as const,
            value: `<strong><span class="smallcaps">${label}</span>${rest}${remainingText}</strong>`,
          }

          parent.children.splice(index, 1, htmlNode)
        })
      },
    ]
  },
})
