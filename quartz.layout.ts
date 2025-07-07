import {PageLayout, SharedLayout} from "./quartz/cfg"
import * as Component from "./quartz/components"
import {FileTrieNode} from "./quartz/util/fileTrie";

// components shared across all pages
export const sharedPageComponents: SharedLayout = {
    head: Component.Head(),
    header: [],
    afterBody: [],
    footer: Component.Footer({
        links: {
            // GitHub: "https://github.com/jackyzha0/quartz",
            // "Discord Community": "https://discord.gg/cRFFHYye7t",
        },
    }),
}

const navOrder  = (a: FileTrieNode, b: FileTrieNode) => {
    const PRIORITY = ["Projects", "Essays"]

    const rank = (n: FileTrieNode) =>
        PRIORITY.findIndex((x) => x.toLowerCase() === n.displayName.toLowerCase())

    const rA = rank(a)
    const rB = rank(b)

    if (rA !== rB) return rA - rB          // honour custom list
    if (a.isFolder !== b.isFolder)         // keep folders above files
        return a.isFolder ? -1 : 1
    return a.displayName.localeCompare(b.displayName) // fallback α-sorting
}


// components for pages that display a single page (e.g. a single note)
export const defaultContentPageLayout: PageLayout = {
    beforeBody: [
        Component.Breadcrumbs(),
        Component.ArticleTitle(),
        Component.ContentMeta({
            showReadingTime: false
        }),
        Component.TagList(),
    ],
    left: [
        Component.PageTitle(),
        Component.MobileOnly(Component.Spacer()),
        Component.Flex({
            components: [
                {
                    Component: Component.Search(),
                    grow: true,
                },
                {Component: Component.Darkmode()},
            ],
        }),
        Component.Explorer({
            folderDefaultState: "open",
            sortFn: navOrder
        }),
    ],
    right: [
        // Component.Graph(),
        Component.ArticleImage(),
        Component.DesktopOnly(Component.TableOfContents()),
        Component.Backlinks(),
    ],
}

// components for pages that display lists of pages  (e.g. tags or folders)
export const defaultListPageLayout: PageLayout = {
    beforeBody: [
        Component.Breadcrumbs(),
        Component.ArticleTitle(),
        Component.ContentMeta({
            showReadingTime: false
        })
    ],
    left: [
        Component.PageTitle(),
        Component.MobileOnly(Component.Spacer()),
        Component.Flex({
            components: [
                {
                    Component: Component.Search(),
                    grow: true,
                },
                {Component: Component.Darkmode()},
            ],
        }),
        Component.Explorer(
        //     {
        //     mapFn: (node) => {
        //         if (!node.isFolder) {
        //             node.displayName = "---jnfvjd " + node.displayName
        //         }
        //     },
        // }
        ),
    ],
    right: [],
}
