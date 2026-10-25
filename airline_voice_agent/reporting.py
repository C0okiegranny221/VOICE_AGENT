"""Save intermediate results and test runs to report/ for the project write-up.

    from reporting import save_artifact, log_entry
    path = save_artifact("parsing", "section_outline", markdown_text)
    log_entry("parsing", "Parsed 5 PDFs", "68 sections in domestic CoC ...", [path])

Each artifact is timestamped so re-runs never overwrite earlier results; report/LOG.md lists every
entry in order. Set REPORT=0 to switch reporting off (e.g. inside quick experiments).
"""
import datetime as dt
import json
import os
from pathlib import Path

REPORT_DIR = Path(__file__).resolve().parent / "report"
STAGES = {
    "data_collection": "01_data_collection",
    "parsing": "02_parsing",
    "chunking": "03_chunking",
    "retrieval": "04_indexing_retrieval",
    "unit_tests": "05_unit_tests",
    "policy_tables": "06_policy_tables",
    "agent": "07_agent_text",
    "voice": "08_voice",
    "eval": "09_eval",
    "calls": "10_calls",
}
LOG = REPORT_DIR / "LOG.md"


def enabled() -> bool:
    return os.getenv("REPORT", "1") != "0"


def stage_dir(stage: str) -> Path:
    d = REPORT_DIR / STAGES[stage]
    d.mkdir(parents=True, exist_ok=True)
    return d


def _stamp() -> str:
    return dt.datetime.now().strftime("%Y-%m-%dT%H%M%S")


def save_artifact(stage: str, name: str, content, ext: str | None = None) -> Path | None:
    """Write content (str, or dict/list -> JSON) to report/<stage>/<timestamp>_<name>.<ext>."""
    if not enabled():
        return None
    if ext is None:
        ext = "json" if isinstance(content, (dict, list)) else "md"
    path = stage_dir(stage) / f"{_stamp()}_{name}.{ext}"
    if isinstance(content, (dict, list)):
        content = json.dumps(content, indent=2, ensure_ascii=False, default=str)
    path.write_text(content)
    return path


def log_entry(stage: str, title: str, summary: str = "", files: list[Path | None] | None = None) -> None:
    """Append one entry to report/LOG.md."""
    if not enabled():
        return
    REPORT_DIR.mkdir(exist_ok=True)
    if not LOG.exists():
        LOG.write_text("# Build log\n\nChronological record of pipeline runs, tests and fixes. "
                       "Artifacts are in the numbered stage folders.\n")
    links = ", ".join(f"[{p.name}]({p.relative_to(REPORT_DIR)})" for p in (files or []) if p)
    when = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    entry = f"\n## {when} · {STAGES[stage]} · {title}\n\n"
    if summary:
        entry += summary.strip() + "\n"
    if links:
        entry += f"\nFiles: {links}\n"
    with LOG.open("a") as f:
        f.write(entry)


def md_table(rows: list[dict], columns: list[str] | None = None) -> str:
    """Render a list of dicts as a Markdown table."""
    if not rows:
        return "_(no rows)_\n"
    columns = columns or list(rows[0])
    out = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")).replace("|", "\\|").replace("\n", " ") for c in columns) + " |")
    return "\n".join(out) + "\n"
