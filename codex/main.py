## Eg. python scripts/run_experiment.py --smell HM

import argparse
import json
import sys

from smell_detection.graph import build_graph

# function to load arguments
def parse_args():
    parser = argparse.ArgumentParser(
        description="Agentic architectural smell detection with Codex."
    )

    # to check if config path is provided
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML configuration file.",
    )

    return parser.parse_args()


def main():
    # load configuration
    args = parse_args()

    # load graph
    app = build_graph()

    try:
        # execute graph
        result = app.invoke(
            {
                "config_path": args.config,
            }
        )
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    print(json.dumps(
        {
            "target": result.get("unit"),
            "smell": result.get("smell"),
            "valid": result.get("valid"),
            "verdict_path": result.get("verdict_path"),
            "error": result.get("error"),
            "codex_returncode": result.get("codex_returncode"),
            "codex_stdout_path": result.get("codex_stdout_path"),
            "codex_stderr_path": result.get("codex_stderr_path"),
            "codex_prompt_path": result.get("codex_prompt_path"),
        },
        indent=2,
        ensure_ascii=False,
    ))

    return 0 if result.get("valid") else 1


if __name__ == "__main__":
    raise SystemExit(main())