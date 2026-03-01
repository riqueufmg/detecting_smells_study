# file: agents/main.py
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from arch_detection import run_folder_with_single_rule, run_one_with_rule, build_single_rule

UD_DEFINITION = (
    "Unstable Dependency arises when a component depends on other less stable components. "
    "Dependencies should point toward stability; a package should only depend on packages more stable than itself."
)

DEFAULT_RULE_PATH = "out/ud_rule.json"


def main() -> None:
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY not found. Put it in .env (OPENAI_API_KEY=...) or export it in your shell."
        )

    parser = argparse.ArgumentParser(description="Run Unstable Dependency detection (single rule).")
    parser.add_argument("--input", required=True, help="Path to a JSON file OR a folder containing JSON files.")
    parser.add_argument("--out", default="out/ud_predictions.jsonl", help="Output path (.json for file, .jsonl for folder).")
    parser.add_argument("--rule", default=DEFAULT_RULE_PATH, help="Path to cache the single rule (json).")
    args = parser.parse_args()

    inp = Path(args.input)
    outp = Path(args.out)
    rule_path = str(Path(args.rule))

    # Build/load ONE rule once
    rule_spec = None
    if Path(rule_path).exists():
        rule_spec = json.loads(Path(rule_path).read_text(encoding="utf-8"))
    else:
        rule_spec = build_single_rule("unstable_dependency", UD_DEFINITION)
        Path(rule_path).parent.mkdir(parents=True, exist_ok=True)
        Path(rule_path).write_text(json.dumps(rule_spec, ensure_ascii=False, indent=2), encoding="utf-8")

    if inp.is_file():
        # single JSON -> single output JSON
        if outp.suffix.lower() != ".json":
            outp = outp.with_suffix(".json")
        result = run_one_with_rule(
            json_path=str(inp),
            smell_type="unstable_dependency",
            smell_definition=UD_DEFINITION,
            rule_spec=rule_spec,
        )
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved: {outp}")

    elif inp.is_dir():
        # folder -> JSONL
        if outp.suffix.lower() != ".jsonl":
            outp = outp.with_suffix(".jsonl")
        run_folder_with_single_rule(
            root_dir=str(inp),
            smell_type="unstable_dependency",
            smell_definition=UD_DEFINITION,
            out_path=str(outp),
            rule_path=rule_path,  # will reuse cached rule
        )
        print(f"Saved: {outp}")

    else:
        raise SystemExit(f"Input not found: {inp}")


if __name__ == "__main__":
    main()