"""Get every source listed in sources.yaml into data/pdfs/<doc_id>.pdf.

PDFs are downloaded as-is; web pages are printed to PDF with the installed Chrome via Playwright.
Writes data/pdfs/<doc_id>.meta.json with source_url, retrieved_on and sha256.
Existing PDFs are skipped unless --force is given.
"""
import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import httpx
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CONFIG  # noqa: E402

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
)


def load_sources() -> list[dict]:
    return yaml.safe_load(CONFIG.sources_file.read_text())


def download_pdf(url: str, out: Path) -> None:
    with httpx.Client(follow_redirects=True, timeout=60, headers={"User-Agent": USER_AGENT}) as client:
        resp = client.get(url)
        resp.raise_for_status()
    if not resp.content.startswith(b"%PDF"):
        raise ValueError(f"{url} did not return a PDF (content-type {resp.headers.get('content-type')})")
    out.write_bytes(resp.content)


# Site chrome that should not end up in the printed policy PDF (goindigo.in AEM components).
HIDE_CSS = """
header, footer, nav, .cc-window, .launcher-anchor, [class*="headerv2"], [class*="footer"],
[class*="breadcrumb"], .cmp-tabs__tablist { display: none !important; }
"""
ACCORDION_HEADER = ".accordion-item--header"
TAB_BUTTON = ".cmp-tabs__tab"


def _print_one(context, src: dict, out: Path) -> None:
    page = context.new_page()
    try:
        page.goto(src["source_url"], wait_until="domcontentloaded", timeout=60_000)
        try:
            page.wait_for_load_state("networkidle", timeout=15_000)
        except Exception:
            pass  # pages with constant background requests never go idle; DOM is enough
        page.wait_for_timeout(3000)
        if src.get("tab"):
            page.locator(TAB_BUTTON, has_text=src["tab"]).first.click()
            page.wait_for_timeout(1500)
        headers = page.locator(ACCORDION_HEADER)
        for i in range(headers.count()):
            h = headers.nth(i)
            if h.is_visible():
                h.click()
                page.wait_for_timeout(150)
        page.add_style_tag(content=HIDE_CSS)
        page.wait_for_timeout(500)
        page.emulate_media(media="screen")
        page.pdf(path=str(out), format="A4", print_background=False)
    finally:
        page.close()


def print_pages(items: list[tuple[dict, Path]], attempts: int = 3) -> list[str]:
    """Print web pages to PDF. Selects src['tab'] if given and expands every accordion first.
    Returns the doc_ids that failed."""
    from playwright.sync_api import sync_playwright

    failed = []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True, args=["--disable-http2"])
        context = browser.new_context(user_agent=USER_AGENT, viewport={"width": 1280, "height": 900})
        for src, out in items:
            for attempt in range(1, attempts + 1):
                try:
                    _print_one(context, src, out)
                    write_meta(src, out)
                    print(f"print  {src['doc_id']}")
                    break
                except Exception as e:
                    print(f"retry  {src['doc_id']} ({attempt}/{attempts}): {str(e).splitlines()[0]}")
            else:
                failed.append(src["doc_id"])
        browser.close()
    return failed


def write_meta(src: dict, pdf: Path) -> None:
    meta = {
        "doc_id": src["doc_id"],
        "title": src["title"],
        "source_url": src.get("source_url"),
        "retrieved_on": dt.date.today().isoformat(),
        "sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
    }
    pdf.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-fetch even if the PDF exists")
    ap.add_argument("--only", nargs="*", help="doc_ids to fetch")
    args = ap.parse_args()

    CONFIG.pdf_dir.mkdir(parents=True, exist_ok=True)
    to_print = []
    for src in load_sources():
        if args.only and src["doc_id"] not in args.only:
            continue
        out = CONFIG.pdf_dir / f"{src['doc_id']}.pdf"
        if out.exists() and not args.force:
            if not out.with_suffix(".meta.json").exists():
                write_meta(src, out)
            print(f"skip   {src['doc_id']} (exists)")
            continue
        if src["get"] == "pdf":
            download_pdf(src["source_url"], out)
            write_meta(src, out)
            print(f"pdf    {src['doc_id']}")
        elif src["get"] == "print":
            to_print.append((src, out))
        elif src["get"] == "manual":
            print(f"MISSING {src['doc_id']}: place the PDF at {out}")

    failed = print_pages(to_print) if to_print else []
    if failed:
        print(f"FAILED {failed}: save these pages as PDF by hand into {CONFIG.pdf_dir}")
    report_inventory(failed)


def report_inventory(failed: list[str] | None = None) -> None:
    """Save a table of every source PDF (pages, characters of text, hash) to the report."""
    import pymupdf

    from reporting import log_entry, md_table, save_artifact

    rows = []
    for src in load_sources():
        pdf = CONFIG.pdf_dir / f"{src['doc_id']}.pdf"
        meta_file = pdf.with_suffix(".meta.json")
        meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
        if pdf.exists():
            doc = pymupdf.open(pdf)
            chars = sum(len(p.get_text()) for p in doc)
            rows.append({"doc_id": src["doc_id"], "method": src["get"], "pages": doc.page_count, "text_chars": chars,
                         "retrieved_on": meta.get("retrieved_on", ""), "sha256": meta.get("sha256", "")[:12],
                         "source_url": src.get("source_url", "")})
        else:
            rows.append({"doc_id": src["doc_id"], "method": src["get"], "pages": "MISSING"})
    table = md_table(rows)
    note = f"\nFailed this run: {failed}\n" if failed else ""
    path = save_artifact("data_collection", "source_inventory", "# Source PDF inventory\n\n" + table + note)
    log_entry("data_collection", f"Source inventory: {sum(1 for r in rows if r['pages'] != 'MISSING')} PDFs", table + note, [path])


if __name__ == "__main__":
    main()
