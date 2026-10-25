"""data/tables/*.csv -> data/policy.db (one SQLite table per CSV).

Every row must carry provenance: source_doc, section, page and a verbatim source_quote. The quote is
checked against the parsed text of that document (data/processed/<doc>.json); a row whose quote is
not found in the named section fails the build. `human_verified_on` records manual review; rows
without it are loaded but reported as pending review.
"""
import csv
import json
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CONFIG  # noqa: E402
from reporting import log_entry, md_table, save_artifact  # noqa: E402

REQUIRED = ["source_doc", "section", "page", "source_quote", "human_verified_on"]


def _norm(text: str) -> str:
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text).strip().lower()


def _sections(doc_id: str) -> list[dict]:
    return json.loads((CONFIG.processed_dir / f"{doc_id}.json").read_text())["sections"]


def check_quote(row: dict) -> str:
    """'ok' if the quote is in the named section, 'other_section' if only elsewhere in the doc, else 'missing'."""
    quote = _norm(row["source_quote"])
    sections = _sections(row["source_doc"])
    named = [s for s in sections if row["section"].lower() in " › ".join(s["heading_path"]).lower()]
    if any(quote in _norm(s["text"]) for s in named):
        return "ok"
    if any(quote in _norm(s["text"]) for s in sections):
        return "other_section"
    return "missing"


def _coerce(value: str):
    if value == "":
        return None
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def build_tables() -> dict:
    db_path = CONFIG.policy_db
    db_path.unlink(missing_ok=True)
    con = sqlite3.connect(db_path)
    results, problems, pending = [], [], 0
    for csv_file in sorted(CONFIG.tables_dir.glob("*.csv")):
        rows = list(csv.DictReader(csv_file.open(encoding="utf-8")))
        table = csv_file.stem
        missing_cols = [c for c in REQUIRED if c not in (rows[0] if rows else {})]
        if missing_cols:
            raise ValueError(f"{csv_file.name}: missing columns {missing_cols}")
        cols = list(rows[0])
        con.execute(f"CREATE TABLE {table} (row_id INTEGER PRIMARY KEY, {', '.join(f'{c}' for c in cols)}, provenance TEXT)")
        for i, row in enumerate(rows, start=2):  # line 1 is the header
            status = check_quote(row)
            if status != "ok":
                problems.append({"table": table, "csv_line": i, "status": status, "quote": row["source_quote"][:80]})
            if not row["human_verified_on"]:
                pending += 1
            con.execute(
                f"INSERT INTO {table} ({', '.join(cols)}, provenance) VALUES ({', '.join('?' * (len(cols) + 1))})",
                [_coerce(row[c]) for c in cols] + [status],
            )
        results.append({"table": table, "rows": len(rows),
                        "quotes_ok": sum(1 for p in problems if p["table"] == table) == 0})
        print(f"table {table}: {len(rows)} rows")
    con.commit()
    con.close()

    total = sum(r["rows"] for r in results)
    summary = (
        f"{len(results)} tables, {total} rows. Source quotes verified against parsed text: "
        f"{total - len(problems)}/{total}. Pending human review: {pending}/{total} rows."
    )
    body = f"# Policy tables build\n\n{summary}\n\n{md_table(results)}"
    if problems:
        body += "\n## Provenance problems\n\n" + md_table(problems)
    snapshot = "\n\n".join(f"## {f.name}\n\n```csv\n{f.read_text().strip()}\n```" for f in sorted(CONFIG.tables_dir.glob("*.csv")))
    files = [save_artifact("policy_tables", "tables_build", body),
             save_artifact("policy_tables", "tables_snapshot", "# Policy table contents at build time\n\n" + snapshot)]
    log_entry("policy_tables", "Built policy.db" + (" — PROVENANCE PROBLEMS" if problems else ""), summary, files)
    if problems:
        print(md_table(problems))
        raise SystemExit(f"{len(problems)} row(s) failed the source-quote check; fix the CSVs (see report).")
    print(summary)
    return {"tables": results, "pending_review": pending}


if __name__ == "__main__":
    build_tables()
