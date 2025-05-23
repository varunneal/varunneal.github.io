import { QuartzTransformerPlugin } from "../types"
import { Root } from "mdast"
import { visit } from "unist-util-visit"

export const CustomPoetry: QuartzTransformerPlugin = () => ({
    name: "CustomPoetry",
    markdownPlugins() {
        return [
            () => (tree: Root) => {
                visit(tree, "code", (node) => {
                    if (!node.lang) return        // no language → let Quartz handle it normally

                    // grab only the first token in case the fence looks like ```ts title=foo
                    const rawLang = node.lang.split(/\s+/)[0]

                    // very light sanitising → lang="c++" ➜ "c--"
                    const langClass = rawLang.replace(/[^a-z0-9-]/gi, "-").toLowerCase()

                    // escape the code so it stays intact inside the HTML we’re about to inject
                    const escaped = node.value
                        .replace(/&/g, "&amp;")
                        .replace(/</g, "&lt;")
                        .replace(/>/g, "&gt;")

                    node.type  = "html" as "code"
                    node.value = `<pre class="codeblock lang-${langClass}"><code>${escaped}</code></pre>`
                })
            },
        ]
    },
})
