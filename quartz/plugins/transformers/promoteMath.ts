import { QuartzTransformerPlugin } from "../types"
import { Root, Paragraph, InlineMath, Math } from "mdast"
import { visit } from "unist-util-visit"

export const PromoteInlineMath: QuartzTransformerPlugin = () => ({
  name: "PromoteInlineMath",
  textTransform(_ctx, src) {
    // Normalize $$<content> at start of line: split $$ onto its own line
    // Skip single-line $$...$$ (which also end with $$)
    src = src.replace(/^\$\$(.+)$/gm, (match, content) => {
      if (/\$\$\s*$/.test(content)) return match
      return "$$\n" + content
    })
    // Normalize <content>$$ at end of line when inside math (line contains \commands)
    // This handles \end{aligned}\right.$$ and similar closing patterns
    src = src.replace(/^(.*\\[a-zA-Z].*)\$\$\s*$/gm, (match, content) => {
      // Don't touch lines that start with $$ (handled above or single-line)
      if (content.startsWith("$$")) return match
      return content + "\n$$"
    })
    return src
  },
  markdownPlugins() {
    return [
      () => (tree: Root) => {
        // Promote single-line $$...$$ (parsed as inline math in a lone paragraph) to display math
        visit(tree, "paragraph", (node: Paragraph, index, parent) => {
          if (!parent || index === undefined) return
          if (node.children.length !== 1) return
          const child = node.children[0]
          if (child.type !== "inlineMath") return

          const value = (child as InlineMath).value
          const mathNode: Math = {
            type: "math",
            value,
            meta: null,
            data: {
              hName: "pre",
              hChildren: [
                {
                  type: "element",
                  tagName: "code",
                  properties: { className: ["language-math", "math-display"] },
                  children: [{ type: "text", value }],
                },
              ],
            },
          }

          parent.children[index] = mathNode as any
        })
      },
    ]
  },
})
