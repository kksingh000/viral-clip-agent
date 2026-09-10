#!/usr/bin/env python3
"""Write several files from one stdin stream.

Input format::

    === relative/path/to/file ===
    <file content>
    === another/file ===
    <file content>

Used by the build tooling so that a batch of files can be created in a single
shell invocation without nesting quotes.
"""
from __future__ import annotations

import ast
import pathlib
import re
import sys

HEADER = re.compile(r"^=== (.+?) ===$")


def main() -> int:
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    text = sys.stdin.read()
    current: pathlib.Path | None = None
    buf: list[str] = []
    written: list[pathlib.Path] = []

    def flush() -> None:
        if current is None:
            return
        current.parent.mkdir(parents=True, exist_ok=True)
        current.write_text("".join(buf), encoding="utf-8", newline="\n")
        written.append(current)

    for line in text.splitlines(keepends=True):
        m = HEADER.match(line.rstrip("\r\n"))
        if m:
            flush()
            current = root / m.group(1).strip()
            buf = []
        elif current is not None:
            buf.append(line)
    flush()

    failures = 0
    for path in written:
        rel = path.relative_to(root)
        note = ""
        if path.suffix == ".py":
            try:
                ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError as exc:
                note = f"  !! SyntaxError line {exc.lineno}: {exc.msg}"
                failures += 1
        print(f"wrote {rel} ({path.stat().st_size} bytes){note}")
    print(f"-- {len(written)} file(s), {failures} syntax error(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
