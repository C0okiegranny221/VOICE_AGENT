"""chunks.jsonl -> LanceDB table "chunks" with a vector column and a full-text (BM25) index.

Rebuilt from scratch on every run.
"""
import json
import sys
from pathlib import Path

import lancedb
from lancedb.index import FTS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CONFIG  # noqa: E402
from rag.embed import embed_passages  # noqa: E402
from reporting import log_entry, save_artifact  # noqa: E402

TABLE = "chunks"


def build_index() -> int:
    chunks = [json.loads(line) for line in (CONFIG.processed_dir / "chunks.jsonl").read_text().splitlines()]
    vectors = embed_passages([c["search_text"] for c in chunks])
    rows = [{**c, "vector": v.tolist()} for c, v in zip(chunks, vectors)]

    db = lancedb.connect(str(CONFIG.lancedb_dir))
    table = db.create_table(TABLE, data=rows, mode="overwrite")
    table.create_index("search_text", config=FTS(), replace=True)
    print(f"indexed {table.count_rows()} chunks into {CONFIG.lancedb_dir}/{TABLE}")
    summary = (
        f"Rows: {table.count_rows()} · embedding model `{CONFIG.embed_model}` (dim {len(rows[0]['vector'])}, "
        f"normalized) · full-text index on `search_text` (BM25, English stemming, stop words removed)."
    )
    log_entry("retrieval", "Built LanceDB index", summary,
              [save_artifact("retrieval", "index_summary", "# Index summary\n\n" + summary)])
    return table.count_rows()


if __name__ == "__main__":
    build_index()
