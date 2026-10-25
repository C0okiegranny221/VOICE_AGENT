"""Save every pytest run to report/05_unit_tests/ (pass/fail per test, durations, failure text)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_results = []


def pytest_runtest_logreport(report):
    if report.when == "call" or (report.when == "setup" and report.outcome != "passed"):
        _results.append(
            {
                "test": report.nodeid,
                "outcome": report.outcome,
                "seconds": round(report.duration, 3),
                "detail": str(report.longrepr)[-1500:] if report.failed else "",
            }
        )


def pytest_sessionfinish(session, exitstatus):
    from reporting import log_entry, md_table, save_artifact

    if not _results:
        return
    passed = sum(r["outcome"] == "passed" for r in _results)
    failed = [r for r in _results if r["outcome"] == "failed"]
    headline = f"{passed}/{len(_results)} passed" + (f", {len(failed)} failed" if failed else "")
    body = f"# Unit test run: {headline}\n\n" + md_table(_results, ["test", "outcome", "seconds"])
    for r in failed:
        body += f"\n## FAILED {r['test']}\n\n```\n{r['detail']}\n```\n"
    path = save_artifact("unit_tests", "pytest_" + ("pass" if not failed else "fail"), body)
    log_entry("unit_tests", f"pytest: {headline}", "", [path])
