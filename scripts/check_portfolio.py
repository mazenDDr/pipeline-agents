"""Verify the static portfolio in installed Google Chrome, not bundled Chromium.

Start: python -m http.server 8765 --directory site
Check: python scripts/check_portfolio.py --out /tmp/pipeline-portfolio
"""

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8765/")
    parser.add_argument("--out", type=Path, default=Path("/tmp/pipeline-portfolio"))
    parser.add_argument("--readme", help="Optional GitHub branch README URL to inspect")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="chrome", headless=True)
        for width, height in ((1440, 900), (390, 844)):
            for route in ("", "field-test/", "tour/"):
                page = browser.new_page(viewport={"width": width, "height": height}, color_scheme="light")
                errors = []
                page.on("pageerror", lambda error, log=errors: log.append(str(error)))
                page.on(
                    "console",
                    lambda message, log=errors: log.append(message.text) if message.type == "error" else None,
                )
                page.goto(args.url.rstrip("/") + "/" + route, wait_until="networkidle")
                page.add_style_tag(content="html { scroll-behavior: auto !important; }")
                assert page.locator("h1").count() == 1
                assert page.locator("table").count() > 0
                assert page.locator("details").count() >= 3
                for item in page.locator("details").all():
                    summary = item.locator("summary").first
                    if item.get_attribute("open") != "":
                        summary.click()
                    assert item.evaluate("element => element.open")
                    summary.click()
                    assert not item.evaluate("element => element.open")
                page.locator("#theme").click()
                assert page.locator("html").get_attribute("data-theme") == "dark"
                page.screenshot(path=str(args.out / f"{route.strip('/') or 'guide'}-{width}-dark.png"))
                page.locator("#theme").click()
                for button in page.get_by_role("button", name="Replay figure").all():
                    for _ in range(2):
                        button.click()
                    assert "replay=" in button.locator("..").locator("img").get_attribute("src")
                page.wait_for_function(
                    "Array.from(document.images).every(image => image.complete && image.naturalWidth > 0)"
                )
                # Deep links are checked against parsed HTML in test_portfolio.py.
                for anchor in page.locator('.map a[href^="#"]').all():
                    anchor.locator("xpath=ancestor::details").evaluate("element => element.open = true")
                    anchor.click()
                    assert page.url.endswith(anchor.get_attribute("href"))
                assert page.evaluate("document.documentElement.scrollWidth - innerWidth") == 0
                page.evaluate("scrollTo(0, 0)")
                page.wait_for_timeout(8500)  # SVG img animations reach their long finished-state hold.
                name = route.strip("/") or "guide"
                page.screenshot(path=str(args.out / f"{name}-{width}.png"))
                for index, figure in enumerate(page.locator("figure").all()):
                    figure.scroll_into_view_if_needed()
                    figure.screenshot(path=str(args.out / f"{name}-{width}-figure-{index}.png"))
                page.emulate_media(reduced_motion="reduce")
                assert page.evaluate("matchMedia('(prefers-reduced-motion: reduce)').matches")
                assert not errors, errors
                print(f"{name} {width}×{height}: populated, interactive, no overflow or console errors")
                page.close()
        if args.readme:
            page = browser.new_page(viewport={"width": 1440, "height": 900})
            page.goto(args.readme, wait_until="domcontentloaded")
            page.locator("article img").first.wait_for()
            page.wait_for_timeout(8500)
            images = page.locator("article img")
            assert images.count() == 6
            assert images.evaluate_all("images => images.every(image => image.naturalWidth > 0)")
            page.screenshot(path=str(args.out / "github-readme.png"))
            page.set_viewport_size({"width": 390, "height": 844})
            page.screenshot(path=str(args.out / "github-readme-phone.png"))
            print("GitHub README: all six figures loaded")
        browser.close()


if __name__ == "__main__":
    main()
