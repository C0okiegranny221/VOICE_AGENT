"""Run the offline pipeline: [fetch] -> parse -> chunk -> index -> policy tables.

    python -m ingest.run_all            # use the PDFs already in data/pdfs/
    python -m ingest.run_all --fetch    # first fetch any sources missing from data/pdfs/
"""
import argparse
import subprocess
import sys
from pathlib import Path

from ingest.build_index import build_index
from ingest.build_tables import build_tables
from ingest.chunk import chunk_all
from ingest.parse import parse_all


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="fetch missing sources before parsing")
    args = ap.parse_args()

    if args.fetch:
        subprocess.run([sys.executable, str(Path(__file__).with_name("fetch.py"))], check=True)
    parse_all()
    chunk_all()
    build_index()
    build_tables()


if __name__ == "__main__":
    main()
