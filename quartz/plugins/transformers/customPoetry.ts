import { QuartzTransformerPlugin } from "../types"
import { Root } from "mdast"
import { visit } from "unist-util-visit"

export const CustomPoetry: QuartzTransformerPlugin = () => ({
    name: "CustomPoetry",
    markdownPlugins() {
        return [
            () => (tree: Root) => {
                visit(tree, "code", (node) => {
                    if (!node.lang) return

                    const lang = node.lang.split(/\s+/)[0]   // e.g. "macondo"

                    // escape dangerous chars
                    const escaped = node.value
                        .replace(/&/g, "&amp;")
                        .replace(/</g, "&lt;")
                        .replace(/>/g, "&gt;")

                    node.type  = "html" as "code"
                    node.value = `<pre class="${lang}">${escaped}</pre>`
                })
            },
        ]
    },
})
