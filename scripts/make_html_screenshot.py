"""Render docs/example-report.html as a PNG screenshot.

Run: .venv/bin/python scripts/make_html_screenshot.py
Output: docs/report-screenshot.png
"""
import asyncio
from pathlib import Path

from playwright.async_api import async_playwright


HTML_PATH = Path(__file__).resolve().parent.parent / "docs" / "example-report.html"
PNG_PATH = Path(__file__).resolve().parent.parent / "docs" / "report-screenshot.png"


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(
            viewport={"width": 1280, "height": 900},
            device_scale_factor=2,
        )
        await page.goto(f"file://{HTML_PATH}")
        # Wait for fonts to settle
        await page.wait_for_load_state("networkidle")
        # Expand the LLM response details so the screenshot shows real content
        await page.evaluate(
            "document.querySelectorAll('details').forEach((d, i) => { if (i < 3) d.open = true; })"
        )
        await page.screenshot(path=str(PNG_PATH), full_page=True)
        await browser.close()
    print(f"Wrote {PNG_PATH} ({PNG_PATH.stat().st_size // 1024} KB)")


asyncio.run(main())
