const observer = new IntersectionObserver((entries) => {
  for (const entry of entries) {
    const slug = entry.target.id
    const tocEntryElement = document.querySelector(`a[data-for="${slug}"]`)
    const windowHeight = entry.rootBounds?.height
    if (windowHeight && tocEntryElement) {
      if (entry.boundingClientRect.y < windowHeight) {
        tocEntryElement.classList.add("in-view")
      } else {
        tocEntryElement.classList.remove("in-view")
      }
    }
  }
})

function toggleToc(this: HTMLElement) {
  this.classList.toggle("collapsed")
  this.setAttribute(
    "aria-expanded",
    this.getAttribute("aria-expanded") === "true" ? "false" : "true",
  )
  const content = this.nextElementSibling as HTMLElement | undefined
  if (!content) return
  content.classList.toggle("collapsed")
}

function setupToc() {
  for (const toc of document.getElementsByClassName("toc")) {
    const button = toc.querySelector(".toc-header")
    const content = toc.querySelector(".toc-content")
    if (!button || !content) return
    button.addEventListener("click", toggleToc)
    window.addCleanup(() => button.removeEventListener("click", toggleToc))
  }
}

function setupDynamicTocHeight() {
  const sidebar = document.querySelector(".sidebar.right") as HTMLElement | null
  if (!sidebar) return

  const toc = sidebar.querySelector(".toc") as HTMLElement | null
  if (!toc) return

  const overflowList = toc.querySelector("ul.overflow") as HTMLElement | null
  if (!overflowList) return

  function updateTocHeight() {
    if (!sidebar || !toc || !overflowList) return

    const tocRect = toc.getBoundingClientRect()
    const tocHeaderHeight = toc.querySelector(".toc-header")?.getBoundingClientRect().height ?? 40

    const bottomBound = window.innerHeight - 32
    const availableForList = bottomBound - tocRect.top - tocHeaderHeight

    if (availableForList > 0) {
      toc.style.flexShrink = "0"
      toc.style.height = `${availableForList + tocHeaderHeight}px`
      overflowList.style.maxHeight = `${Math.max(availableForList, 100)}px`
    }
  }

  updateTocHeight()
  window.addEventListener("scroll", updateTocHeight, { passive: true })
  window.addEventListener("resize", updateTocHeight, { passive: true })
  window.addCleanup(() => {
    window.removeEventListener("scroll", updateTocHeight)
    window.removeEventListener("resize", updateTocHeight)
  })
}

document.addEventListener("nav", () => {
  setupToc()
  setupDynamicTocHeight()

  // update toc entry highlighting
  observer.disconnect()
  const headers = document.querySelectorAll("h1[id], h2[id], h3[id], h4[id], h5[id], h6[id]")
  headers.forEach((header) => observer.observe(header))
})
