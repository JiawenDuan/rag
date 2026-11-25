"""Reproduce the six manual reference-line lookups used in the Group 21 demo."""

import argparse
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.parse import unquote, urlsplit

import config
import rag


@dataclass(frozen=True)
class Case:
    project: str
    question: str
    path: str
    line: int
    literal: str


CASES = [
    Case("Tasks", "Which status types does the Tasks ‘not done’ filter include?",
         "docs/Queries/Filters.md", 285, "ON_HOLD"),
    Case("Dataview", "How do I hide the default File column in a Dataview table?",
         "docs/docs/queries/query-types.md", 285, "TABLE WITHOUT ID"),
    Case("Obsidian", "How do I link to a heading in another note?",
         "en/Linking notes and files/Internal links.md", 76, "another note"),
    Case("Obsidian", "Which property types can a note have?",
         "en/Editing and formatting/Properties.md", 42, "Checkbox"),
    Case("Dataview", "How do I sort a Dataview query in descending order?",
         "docs/docs/queries/data-commands.md", 55, "DESC"),
    Case("Tasks", "怎么让重复任务根据完成日期计算下一次日期？",
         "docs/Getting Started/Recurring Tasks.md", 92, "when done"),
]


@lru_cache(maxsize=64)
def snapshot_lines(root: Path, revision: str, name: str) -> list[str]:
    blob = subprocess.check_output(["git", "-C", str(root), "cat-file", "blob", f"{revision}:{name}"])
    return blob.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n").split("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true", help="Return a failure if any reference is missed.")
    args = parser.parse_args()
    docs = {source["project"]: source for source in config.SOURCES}
    passed = 0
    checked = 0
    for case in CASES:
        src = docs[case.project]
        root = config.DATA_DIR / src["dir"]
        target = snapshot_lines(root, src["revision"], case.path)
        if case.literal not in target[case.line - 1]:
            raise ValueError(f"Reference changed: {case.path}:{case.line}. Run prepare_docs.py.")
        result = rag.retrieve(case.question, [case.project], 3)
        match = None
        for hit in result:
            assert hit.project == case.project, "Project filter failed"
            parsed = urlsplit(hit.url)
            expected = f"/{src['repo']}/blob/{src['revision']}/{hit.source}"
            assert parsed.netloc == "github.com" and unquote(parsed.path) == expected, hit.url
            assert parsed.fragment == f"L{hit.start_line}-L{hit.end_line}", hit.url
            original = snapshot_lines(root, src["revision"], hit.source)
            assert hit.raw_content == "\n".join(original[hit.start_line - 1:hit.end_line])
            checked += 1
            if hit.source == case.path and hit.start_line <= case.line <= hit.end_line:
                match = hit.index
        print(f"{'HIT ' + str(match) if match else 'MISS ':6}  {case.project:8} {case.question}", flush=True)
        passed += match is not None
    print(f"Reference line in top 3: {passed}/{len(CASES)}. Verified {checked} returned line ranges.")
    if args.strict and passed != len(CASES):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
