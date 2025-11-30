// Gradually scroll right sidebar content off-screen as user scrolls
// Stop when TOC would start to disappear off the top
const MIN_TOP = 16 // minimum distance from viewport top for TOC

function updateSidebarOffset() {
  const sidebar = document.querySelector(".sidebar.right") as HTMLElement | null
  if (!sidebar) return

  const toc = sidebar.querySelector(".toc") as HTMLElement | null

  // Calculate the max we can shift the sidebar up
  // This is limited by keeping the TOC visible
  let maxShift = 0

  if (toc) {
    // TOC's offset from the top of the sidebar (includes padding)
    const tocOffsetInSidebar = toc.offsetTop
    // We can shift up until TOC is at MIN_TOP from viewport top
    maxShift = Math.max(0, tocOffsetInSidebar - MIN_TOP)
  }

  const scrollY = window.scrollY
  // Shift up 1:1 with scroll, but cap at maxShift
  const shift = Math.min(scrollY, maxShift)

  // Use negative top value to pull sidebar up
  sidebar.style.top = `${-shift}px`
}

document.addEventListener("nav", () => {
  // Reset on navigation
  const sidebar = document.querySelector(".sidebar.right") as HTMLElement | null
  if (sidebar) {
    sidebar.style.top = "0px"
  }

  window.addEventListener("scroll", updateSidebarOffset, { passive: true })
  window.addCleanup(() => window.removeEventListener("scroll", updateSidebarOffset))

  // Initial update in case page loads scrolled
  updateSidebarOffset()
})
