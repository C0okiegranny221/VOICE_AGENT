"""PDF -> sections with a heading path and page number.

Output: data/processed/<doc_id>.json = {"doc": {...meta}, "sections": [{"heading_path", "page", "text"}]}

Headings are found from structure rather than fonts (the printed IndiGo pages use one font for
everything). sources.yaml `headings:` picks the level-1 rule per document:
  numbered  (default): "6. Tariff, Taxes and Charges" - must follow the previous level-1 number in
            sequence, which stops numbered lists inside a clause from being taken as headings.
  accordion: a line followed by an icon glyph (Unicode private-use area) - the expand arrow that
            goindigo.in prints after every accordion title. Numbered or not.
Level 2 is "6.1 Tariff" or "1.5" / "1.6 This CAR is ...", whose major must equal the current
level-1 number. Under an unnumbered accordion title, sequential "1. Title" lines are level 2.
"""
import collections
import json
import re
import sys
from pathlib import Path

import pymupdf
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CONFIG  # noqa: E402
from reporting import log_entry, md_table, save_artifact  # noqa: E402

L1_RE = re.compile(r"^(\d{1,2})\.\s+([A-Z].{1,90})$")
L2_RE = re.compile(r"^(\d{1,2})\.(\d{1,2})\.?(?:\s+(.*))?$")
MIN_CHARS_PER_PAGE = 200


def _norm(line: str) -> str:
    return re.sub(r"\d+", "#", line.strip().lower())


def _is_icon(line: str) -> bool:
    return bool(line) and all(0xE000 <= ord(c) <= 0xF8FF for c in line)


def extract_lines(pdf: Path) -> list[tuple[int, str, bool]]:
    """(page_number, line, followed_by_icon) for every text line, minus repeated headers/footers,
    page numbers and icon glyphs."""
    doc = pymupdf.open(pdf)
    pages = []
    for page in doc:
        lines = [ln.replace("\xa0", " ").strip() for ln in page.get_text("text").splitlines()]
        pages.append([ln for ln in lines if ln])

    # Lines that recur on most pages are running headers/footers (e.g. "Rev. 4, dated 25th Jan, 2023").
    counts = collections.Counter(n for lines in pages for n in {_norm(ln) for ln in lines})
    repeated = {n for n, c in counts.items() if len(pages) >= 3 and c >= 0.5 * len(pages)}

    flat = [(pno, ln) for pno, lines in enumerate(pages, start=1) for ln in lines]
    out = []
    for i, (pno, ln) in enumerate(flat):
        if _is_icon(ln) or _norm(ln) in repeated or re.fullmatch(r"\d{1,3}", ln):
            continue
        followed_by_icon = i + 1 < len(flat) and _is_icon(flat[i + 1][1])
        out.append((pno, ln, followed_by_icon))
    total_chars = sum(len(ln) for _, ln, _ in out)
    if total_chars < MIN_CHARS_PER_PAGE * max(1, len(pages)) / 4:
        print(f"WARNING {pdf.name}: very little text ({total_chars} chars) - scanned PDF? OCR is out of scope.")
    return out


def _is_title(text: str, allow_period: bool = False) -> bool:
    """Short and not a sentence -> a clause title rather than the start of body text."""
    endings = (",", ";") if allow_period else (".", ",", ";")
    return bool(text) and len(text) <= 70 and not text.rstrip().endswith(endings)


def split_sections(lines: list[tuple[int, str, bool]], doc_title: str, mode: str = "numbered") -> list[dict]:
    sections = []
    cur = {"heading_path": [doc_title], "page": lines[0][0] if lines else 1, "lines": []}
    l1_num, l1_title, l2_num = 0, None, 0

    def flush():
        if cur["lines"]:
            sections.append({"heading_path": cur["heading_path"], "page": cur["page"], "text": _join(cur["lines"])})

    for page, ln, marked in lines:
        m1 = L1_RE.match(ln)
        if mode == "accordion":
            is_l1 = marked
        else:
            # Level-1 titles may end with "." ("5. Bookings - ... & Invoices."); the sequence check guards them.
            is_l1 = bool(m1) and int(m1.group(1)) == l1_num + 1 and _is_title(m1.group(2), allow_period=True)
        if is_l1:
            flush()
            l1_num = int(m1.group(1)) if m1 else 0
            l1_title = f"{l1_num}. {m1.group(2).strip().rstrip('.:')}" if m1 else ln.rstrip(".:")
            l2_num = 0
            cur = {"heading_path": [doc_title, l1_title], "page": page, "lines": []}
            continue

        m2 = L2_RE.match(ln)
        if m2 and l1_num and int(m2.group(1)) == l1_num and int(m2.group(2)) > l2_num:
            l2_num = int(m2.group(2))
            rest = (m2.group(3) or "").strip()
            label = f"{l1_num}.{l2_num}"
            if _is_title(rest):
                label, body = f"{label} {rest}", []
            else:
                body = [rest] if rest else []
        elif m1 and l1_title and not l1_num and int(m1.group(1)) == l2_num + 1 and _is_title(m1.group(2), True):
            # Unnumbered accordion title (e.g. a region) with "1. ...", "2. ..." sub-clauses.
            l2_num = int(m1.group(1))
            label, body = f"{l2_num}. {m1.group(2).strip().rstrip('.:')}", []
        else:
            cur["lines"].append(ln)
            continue
        flush()
        cur = {"heading_path": [doc_title, l1_title, label], "page": page, "lines": body}
    flush()
    return sections


def _join(lines: list[str]) -> str:
    """Undo PDF line wrapping: keep a newline only where a line ends a sentence/item."""
    text = ""
    for ln in lines:
        if not text:
            text = ln
        elif text.endswith((".", ":", ";", "?")) or re.match(r"^([a-z]\)|\([a-z]+\)|[ivx]+\)|[•\-–])\s", ln):
            text += "\n" + ln
        else:
            text += " " + ln
    return re.sub(r"[ \t]+", " ", text).strip()


def parse_all() -> list[Path]:
    sources = yaml.safe_load(CONFIG.sources_file.read_text())
    CONFIG.processed_dir.mkdir(parents=True, exist_ok=True)
    written, summary, outlines = [], [], []
    for src in sources:
        pdf = CONFIG.pdf_dir / f"{src['doc_id']}.pdf"
        if not pdf.exists():
            print(f"MISSING {pdf.name} - skipped")
            continue
        meta_file = pdf.with_suffix(".meta.json")
        meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
        doc = {
            "doc_id": src["doc_id"],
            "title": src["title"],
            "source_url": src.get("source_url"),
            "retrieved_on": meta.get("retrieved_on"),
            "route_scope": src.get("route_scope", "both"),
            "jurisdiction": src.get("jurisdiction", "india"),
        }
        sections = split_sections(extract_lines(pdf), src["title"], src.get("headings", "numbered"))
        out = CONFIG.processed_dir / f"{src['doc_id']}.json"
        out.write_text(json.dumps({"doc": doc, "sections": sections}, indent=1, ensure_ascii=False))
        print(f"parsed {src['doc_id']}: {len(sections)} sections")
        written.append(out)
        summary.append({"doc_id": src["doc_id"], "headings": src.get("headings", "numbered"),
                        "pages": pymupdf.open(pdf).page_count, "sections": len(sections),
                        "max_section_chars": max((len(s["text"]) for s in sections), default=0)})
        outlines.append(f"## {src['doc_id']}\n\n" + md_table(
            [{"page": s["page"], "chars": len(s["text"]), "heading": " › ".join(s["heading_path"][1:]) or "(preamble)"}
             for s in sections]))
    _report(summary, outlines)
    return written


def _report(summary: list[dict], outlines: list[str]) -> None:
    table = md_table(summary)
    outline_file = save_artifact("parsing", "section_outlines", "# Section outlines\n\n" + "\n".join(outlines))
    summary_file = save_artifact("parsing", "parse_summary", "# Parse summary\n\n" + table)
    log_entry("parsing", "Parsed PDFs into sections", table, [summary_file, outline_file])


if __name__ == "__main__":
    parse_all()
