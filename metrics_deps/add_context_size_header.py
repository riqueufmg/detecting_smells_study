from __future__ import annotations

import argparse
from pathlib import Path
import sys

import tiktoken


HEADER_PREFIX = "##CONTEXT_SIZE="


def compute_tokens(text: str, enc) -> int:
    # count tokens using tiktoken encoding
    return len(enc.encode(text))


def normalize_newlines(s: str) -> str:
    # keep content as-is; do not normalize globally
    return s


def update_file(path: Path, enc) -> tuple[bool, int]:
    """
    Returns: (changed, context_size)
    - changed: whether file content was modified
    - context_size: computed token count (excluding header if present)
    """
    raw = path.read_text(encoding="utf-8", errors="replace")

    # Split lines while keeping line endings
    lines = raw.splitlines(keepends=True)

    # If file is empty, treat as empty prompt
    if not lines:
        context_size = compute_tokens("", enc)
        new_first_line = f"{HEADER_PREFIX}{context_size}\n"
        path.write_text(new_first_line, encoding="utf-8")
        return True, context_size

    # Detect and remove existing header line if present in first line
    first_line_stripped = lines[0].rstrip("\r\n")
    has_header = first_line_stripped.startswith(HEADER_PREFIX)

    # Build the prompt text to measure (exclude header if it exists)
    if has_header:
        prompt_text = "".join(lines[1:])
    else:
        prompt_text = "".join(lines)

    prompt_text = normalize_newlines(prompt_text)
    context_size = compute_tokens(prompt_text, enc)

    new_header_line = f"{HEADER_PREFIX}{context_size}\n"

    # Rebuild file
    if has_header:
        # Replace header
        new_lines = [new_header_line] + lines[1:]
        changed = (lines[0] != new_header_line)
    else:
        # Insert header
        new_lines = [new_header_line] + lines
        changed = True

    if changed:
        path.write_text("".join(new_lines), encoding="utf-8")

    return changed, context_size


def iter_txt_files(root: Path):
    yield from root.rglob("*.txt")


def main():
    parser = argparse.ArgumentParser(
        description="Add/refresh ##CONTEXT_SIZE=<tokens> header in prompt .txt files using tiktoken o200k_base."
    )
    parser.add_argument(
        "--root",
        type=str,
        default="data/processed/prompts/smell_detection",
        help="Root directory containing smell_name/repo_name folders.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and report, but do not modify files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional: process only first N files (0 = all).",
    )

    args = parser.parse_args()
    root = Path(args.root)

    if not root.exists():
        print(f"[ERR] Root not found: {root}", file=sys.stderr)
        sys.exit(1)

    encoding = tiktoken.get_encoding("o200k_base")

    total = 0
    changed_count = 0
    skipped_count = 0

    for path in iter_txt_files(root):
        if args.limit and total >= args.limit:
            break

        total += 1

        if args.dry_run:
            raw = path.read_text(encoding="utf-8", errors="replace")
            lines = raw.splitlines(keepends=True)
            if lines and lines[0].rstrip("\r\n").startswith(HEADER_PREFIX):
                prompt_text = "".join(lines[1:])
            else:
                prompt_text = raw
            ctx = compute_tokens(prompt_text, encoding)
            print(f"[DRY] {path} -> {ctx}")
            continue

        try:
            changed, ctx = update_file(path, encoding)
            if changed:
                changed_count += 1
                print(f"[OK]  {path} -> {ctx}")
            else:
                skipped_count += 1
                print(f"[SKIP]{path} (already correct) -> {ctx}")
        except Exception as e:
            print(f"[ERR] {path}: {e}", file=sys.stderr)

    if args.dry_run:
        print(f"\nDone (dry-run). Files scanned: {total}")
    else:
        print(f"\nDone. Files scanned: {total} | changed: {changed_count} | unchanged: {skipped_count}")


if __name__ == "__main__":
    main()
