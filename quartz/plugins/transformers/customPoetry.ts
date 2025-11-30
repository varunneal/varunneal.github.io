import { QuartzTransformerPlugin } from "../types"
import { Root } from "mdast"
import { visit } from "unist-util-visit"

export const CustomPoetry: QuartzTransformerPlugin = () => ({
    name: "CustomPoetry",
    markdownPlugins() {
        return [
            () => (tree: Root) => {
                // Skip syntax highlighting languages - let the syntax highlighter handle these
                const skipLangs = new Set([
                    "python", "javascript", "typescript", "java", "cpp", "c", "rust",
                    "go", "ruby", "php", "swift", "kotlin", "scala", "r", "sql",
                    "html", "css", "scss", "json", "yaml", "xml", "markdown", "md",
                    "bash", "sh", "shell", "powershell", "dockerfile"
                ])

                visit(tree, "code", (node) => {
                    if (!node.lang) return

                    const lang = node.lang.split(/\s+/)[0]   // e.g. "macondo"

                    // Skip programming languages - let syntax highlighter handle them
                    if (skipLangs.has(lang.toLowerCase())) return

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
