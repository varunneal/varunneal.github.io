import { QuartzTransformerPlugin } from "../types"
import { Root } from "mdast"
import { Root as HastRoot, Element, Text } from "hast"
import { visit } from "unist-util-visit"

function getTextContent(node: Element | Text): string {
    if (node.type === "text") return node.value
    return (node.children || [])
        .map((c) => (c.type === "text" ? c.value : c.type === "element" ? getTextContent(c) : ""))
        .join("")
}

export const CustomPoetry: QuartzTransformerPlugin = () => ({
    name: "CustomPoetry",
    markdownPlugins() {
        return [
            () => (tree: Root) => {
                const skipLangs = new Set([
                    "python", "javascript", "typescript", "java", "cpp", "c", "rust",
                    "go", "ruby", "php", "swift", "kotlin", "scala", "r", "sql",
                    "html", "css", "scss", "json", "yaml", "xml", "markdown", "md",
                    "bash", "sh", "shell", "powershell", "dockerfile"
                ])

                visit(tree, "code", (node) => {
                    if (!node.lang) return

                    const lang = node.lang
                    const meta = (node.meta || "").trim()
                    const isHidden = meta === "hidden" || meta.includes("hidden")

                    // Mark hidden code blocks with a sentinel first line
                    if (isHidden) {
                        node.meta = meta.replace(/hidden/g, "").trim() || null
                        node.value = `#__HIDDEN__\n` + node.value
                    }

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
    htmlPlugins() {
        return [
            () => (tree: HastRoot) => {
                visit(tree, "element", (node: Element, index, parent) => {
                    if (!parent || index === undefined) return
                    if (node.tagName !== "figure") return

                    // Find <pre> > <code> inside the figure
                    const pre = node.children.find(
                        (c): c is Element => c.type === "element" && c.tagName === "pre"
                    )
                    if (!pre) return
                    const code = pre.children.find(
                        (c): c is Element => c.type === "element" && c.tagName === "code"
                    )
                    if (!code) return

                    // Check first line for __HIDDEN__ marker
                    const firstLine = code.children.find(
                        (c): c is Element => c.type === "element" && c.tagName === "span"
                    )
                    if (!firstLine) return

                    const firstLineText = getTextContent(firstLine)
                    if (!firstLineText.includes("#__HIDDEN__")) return

                    // Remove the marker line
                    const lineIndex = code.children.indexOf(firstLine)
                    code.children.splice(lineIndex, 1)

                    // Wrap in <details>
                    const details: Element = {
                        type: "element",
                        tagName: "details",
                        properties: { className: ["code-collapse"] },
                        children: [
                            {
                                type: "element",
                                tagName: "summary",
                                properties: {},
                                children: [],
                            },
                            node,
                        ],
                    }

                    parent.children[index] = details
                })
            },
        ]
    },
})
