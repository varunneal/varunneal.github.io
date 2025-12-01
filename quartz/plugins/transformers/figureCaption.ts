import { QuartzTransformerPlugin } from "../types"
import { Root, Element, Text } from "hast"
import { visit } from "unist-util-visit"
import { toString } from "hast-util-to-string"

export const FigureCaption: QuartzTransformerPlugin = () => ({
  name: "FigureCaption",
  htmlPlugins() {
    return [
      () => (tree: Root) => {
        visit(tree, "element", (node: Element, index, parent) => {
          if (!parent || index === undefined) return
          if (node.tagName !== "strong") return

          // Get the text content to check for Figure/Algorithm pattern
          const textContent = toString(node)
          const match = textContent.match(/^(Figure\s+\d+|Algorithm\s+\d+)(:)?/i)
          if (!match) return

          const label = match[1] // "Figure 1" or "Algorithm 1"
          const hasColon = match[2] === ":"

          // Create smallcaps span for the label
          const firstLetter = label.charAt(0)
          const restOfLabel = label.slice(1)
          const words = restOfLabel.split(" ")
          const lastWord = words.pop() || ""
          const middlePart = words.join(" ")

          const smallcapsSpan: Element = {
            type: "element",
            tagName: "span",
            properties: { className: ["smallcaps"] },
            children: [
              {
                type: "element",
                tagName: "span",
                properties: { className: ["cap"] },
                children: [{ type: "text", value: firstLetter }],
              },
              { type: "text", value: middlePart + (middlePart ? " " : "") },
              {
                type: "element",
                tagName: "span",
                properties: { className: ["cap"] },
                children: [{ type: "text", value: lastWord }],
              },
            ],
          }

          // Find and modify the first text node to remove the label part
          let labelRemoved = false
          const removeLabel = (children: (Element | Text | any)[]): void => {
            for (let i = 0; i < children.length; i++) {
              const child = children[i]
              if (child.type === "text" && !labelRemoved) {
                const labelPattern = new RegExp(`^${label.replace(/\s+/g, "\\s+")}:?\\s*`, "i")
                if (labelPattern.test(child.value)) {
                  child.value = child.value.replace(labelPattern, "")
                  labelRemoved = true
                  // If text node is now empty, we could remove it, but leaving it is fine
                  break
                }
              } else if (child.type === "element" && child.children) {
                removeLabel(child.children)
                if (labelRemoved) break
              }
            }
          }

          removeLabel(node.children)

          // Prepend the smallcaps label to the strong's children
          const colonNode: Text | null = hasColon ? { type: "text", value: ": " } : null
          const newChildren = colonNode
            ? [smallcapsSpan, colonNode, ...node.children]
            : [smallcapsSpan, ...node.children]

          node.children = newChildren
        })
      },
    ]
  },
})
