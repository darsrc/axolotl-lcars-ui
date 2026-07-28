const manifestPromise = fetch("/lcars/manifest", {
  credentials: "same-origin",
  headers: { Accept: "application/json" },
}).then((response) => {
  if (!response.ok) {
    throw new Error(`Manifest request failed with ${response.status}`);
  }
  return response.json();
});

async function switchLcarsPage(pageId, fallbackUrl) {
  try {
    const manifest = await manifestPromise;
    const items = manifest?.layout?.sidebar?.items ?? [];
    const pageIndex = items.findIndex((item) => item.target_page === pageId);
    const buttons = document.querySelectorAll(".lcars-rail-btn");
    const button = pageIndex >= 0 ? buttons.item(pageIndex) : null;
    if (button instanceof HTMLButtonElement) {
      button.click();
      button.focus({ preventScroll: true });
      return;
    }
  } catch {
    // Preserve ordinary link behavior if the manifest or navigation rail is unavailable.
  }
  window.location.assign(fallbackUrl);
}

document.addEventListener(
  "click",
  (event) => {
    if (
      event.defaultPrevented ||
      event.button !== 0 ||
      event.metaKey ||
      event.ctrlKey ||
      event.shiftKey ||
      event.altKey ||
      !(event.target instanceof Element)
    ) {
      return;
    }

    const anchor = event.target.closest("a[href]");
    if (!(anchor instanceof HTMLAnchorElement) || anchor.target === "_blank" || anchor.download) {
      return;
    }

    const url = new URL(anchor.href, window.location.href);
    const pageId = url.searchParams.get("page");
    if (url.origin !== window.location.origin || !pageId) {
      return;
    }

    event.preventDefault();
    void switchLcarsPage(pageId, url.href);
  },
  true,
);
