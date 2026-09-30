import { QuartzConfig } from "./quartz/cfg"
import * as Plugin from "./quartz/plugins"

/**
 * Quartz 4 Configuration
 *
 * See https://quartz.jzhao.xyz/configuration for more information.
 */
const config: QuartzConfig = {
  configuration: {
    pageTitle: "@varunneal",
    pageTitleSuffix: "",
    enableSPA: true,
    enablePopovers: true,
    analytics: {
      provider: "google",
      tagId: 'G-X06SGTCDRZ'
    },
    locale: "en-US",
    baseUrl: "varunneal.github.io",
    ignorePatterns: ["private", "templates", ".obsidian"],
    defaultDateType: "created",
    theme: {
      fontOrigin: "googleFonts",
      cdnCaching: true,
      typography: {
        header: "Kumbh Sans",
        body: "Jacques Francois",
        code: "IBM Plex Mono",
      },
      colors: {
        lightMode: {
          light: "#fcfaf3",
          lightgray: "#e6dfd0",
          gray: "#b3ab9d",
          darkgray: "#6a5a4a",
          dark: "#5b493a",
          secondary: "#556b2f",
          tertiary: "#4d6b8a",
          highlight: "#93ad7a",
          textHighlight: "#718953",
        },
        darkMode: {
          light: "#161412",
          lightgray: "#302b26",
          gray: "#5a5047",
          darkgray: "#a99e91",
          dark: "#dcd3c5",
          secondary: "#93ad7a",
          tertiary: "#6c96c2",
          highlight: "#93ad7a",
          textHighlight: "#b0c29d",
        },


      },
    },
  },
  plugins: {
    transformers: [
      Plugin.FrontMatter(),
      Plugin.CreatedModifiedDate({
        priority: ["frontmatter", "git", "filesystem"],
      }),
      Plugin.SyntaxHighlighting({
        theme: {
          light: "github-light",
          dark: "github-dark",
        },
        keepBackground: false,
      }),
      Plugin.ObsidianFlavoredMarkdown({
        enableInHtmlEmbed: false,
        parseArrows: false,
        mermaid: false
      }),
      Plugin.GitHubFlavoredMarkdown(),
      Plugin.TableOfContents({ maxDepth: 4 }),
      Plugin.CrawlLinks({ markdownLinkResolution: "shortest" }),
      Plugin.Description(),
      Plugin.Latex({ renderEngine: "katex" }),
      Plugin.PromoteInlineMath(),
      Plugin.CustomPoetry(),
      Plugin.FigureCaption(),
    ],
    filters: [Plugin.RemoveDrafts()],
    emitters: [
      Plugin.AliasRedirects(),
      Plugin.ComponentResources(),
      Plugin.ContentPage(),
      Plugin.FolderPage(),
      Plugin.TagPage(),
      Plugin.ContentIndex({
        enableSiteMap: true,
        enableRSS: true,
      }),
      Plugin.Assets(),
      Plugin.Static(),
      Plugin.RawContent(),
      Plugin.NotFoundPage(),
      // Comment out CustomOgImages to speed up build time
      // Plugin.CustomOgImages(),
    ],
  },
}

export default config
