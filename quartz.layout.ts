import {PageLayout, SharedLayout} from "./quartz/cfg"
import * as Component from "./quartz/components"
import {FileTrieNode} from "./quartz/util/fileTrie";



function navOrder(a: FileTrieNode, b: FileTrieNode): number {
    // sorted ascending
     const ORDER: Record<string, number>  = {
        'projects': 0,
        'essays': 1
     }

    const rA = ORDER[a.displayName.toLowerCase()] ?? 100
    const rB = ORDER[b.displayName.toLowerCase()] ?? 100

    if (rA !== rB) {
        return rA - rB
    }
    if (a.isFolder !== b.isFolder)
        return a.isFolder ? -1 : 1

    return a.displayName.localeCompare(b.displayName)
}

// components shared across all pages
export const sharedPageComponents: SharedLayout = {
    head: Component.Head(),
    header: [Component.Darkmode()],
    afterBody: [],
    left: [
        // Component.PageTitle(),
        // Component.MobileOnly(Component.Spacer()),
        // Component.Flex({
        //     components: [
        //         {
        //             Component: Component.Search(),
        //             grow: true,
        //         },
        //         {Component: Component.Darkmode()}
        //     ],
        //     gap: "0.4rem"
        // }),
        // Component.Explorer({
        //     folderDefaultState: "open",
        //     sortFn: navOrder
        // }),
    ],
    footer: Component.Footer({
        links: {
            // Resume: "https://varunneal.github.io/resume.pdf",
            Github: "https://github.com/varunneal",
            LinkedIn: "https://www.linkedin.com/in/varun-n-sri/",
            // Spotify: "https://open.spotify.com/user/varun2k",
            Twitter: "https://x.com/varunneal/",
            Email: "mailto:varun.neal@berkeley.edu"
        },
    }),
}

// components for pages that display a single page (e.g. a single note)
export const defaultContentPageLayout: PageLayout = {
    beforeBody: [
        Component.Breadcrumbs(),
        Component.ArticleTitle(),
        // Component.Flex({
        //     components: [
        //         {Component: Component.ArticleTitle(), grow: true},
        //         {Component: Component.Darkmode(), justify: "end"}
        //     ],
        // }),
        Component.ContentMeta({
            showReadingTime: false
        }),
        Component.TagList(),
    ],
    right: [
        // Component.Graph(),
        Component.ArticleImage(),
        Component.Fractal({
            height: 289,
            width: 289
        }),
        Component.DesktopOnly(Component.TableOfContents()),
        // Component.Backlinks(),
    ],
}

// components for pages that display lists of pages  (e.g. tags or folders)
export const defaultListPageLayout: PageLayout = {
    beforeBody: [
        Component.Breadcrumbs(),
    ],
    // afterBody: [
    //     Component.ListPageImage(),
    // ],
    right: [],
}
