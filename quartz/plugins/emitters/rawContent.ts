import { QuartzEmitterPlugin } from "../types"
import { write } from "./helpers"
import { FilePath } from "../../util/path"

export const RawContent: QuartzEmitterPlugin = () => {
  return {
    name: "RawContent",
    getQuartzComponents() {
      return []
    },
    async *emit(ctx, content, _resources) {
      for (const [_tree, file] of content) {
        const slug = file.data.slug!
        if (slug.endsWith("/index") || slug.startsWith("tags/")) continue

        const raw = file.data.rawMarkdown
        if (!raw) continue

        yield write({
          ctx,
          content: raw,
          slug,
          ext: ".md",
        })
      }
    },
  }
}
