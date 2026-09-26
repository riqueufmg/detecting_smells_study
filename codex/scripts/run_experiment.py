import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


SMELL_TO_CSV = {
    "GC": "gc.csv",
    "UD": "ud.csv",
    "IM": "im.csv",
    "HM": "hm.csv",
}


SMELL_TO_FULL_NAME = {
    "GC": "God Component",
    "UD": "Unstable Dependency",
    "IM": "Insufficient Modularization",
    "HM": "Hub-like Modularization",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Codex-based smell detection experiment for multiple samples."
    )

    parser.add_argument(
        "--smell",
        required=True,
        choices=["GC", "UD", "IM", "HM"],
        help="Smell type to execute: GC, UD, IM, or HM.",
    )

    parser.add_argument(
        "--base-config",
        default="config/detect.yml",
        help="Path to the base YAML configuration file.",
    )

    parser.add_argument(
        "--samples-dir",
        default="scripts/samples",
        help="Directory containing sample CSV files.",
    )

    parser.add_argument(
        "--generated-configs-dir",
        default="data/outputs/generated_configs",
        help="Directory where generated YAML configs will be saved.",
    )

    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop the experiment if one sample fails.",
    )

    return parser.parse_args()


def load_base_config(base_config_path: Path) -> dict:
    if not base_config_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_config_path}")

    with base_config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Base config must be a YAML object.")

    return config


def load_samples(samples_path: Path) -> list[dict]:
    if not samples_path.exists():
        raise FileNotFoundError(f"Samples file not found: {samples_path}")

    with samples_path.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        samples = list(reader)

    required_columns = {"project", "path", "smell", "target"}
    found_columns = set(reader.fieldnames or [])

    missing = required_columns - found_columns
    if missing:
        raise ValueError(
            f"Samples file {samples_path} is missing columns: {sorted(missing)}"
        )

    return samples


def safe_name(value: str) -> str:
    return (
        value.replace("/", "_")
        .replace("\\", "_")
        .replace(".", "_")
        .replace("$", "_")
        .replace(" ", "_")
        .replace("-", "_")
    )


def build_sample_config(
    base_config: dict,
    smell_code: str,
    sample: dict,
    index: int,
) -> dict:
    config = dict(base_config)

    # Avoid mutating nested dictionaries from the original config
    config["experiment"] = dict(base_config.get("experiment", {}))
    config["task"] = dict(base_config.get("task", {}))

    project = sample["project"].strip()
    repo_path = sample["path"].strip()
    target = sample["target"].strip()

    config["experiment"]["name"] = f"{project}-{smell_code}-{index:04d}"

    config["task"]["repo_path"] = repo_path
    config["task"]["smell"] = smell_code
    config["task"]["unit"] = target

    return config


def write_generated_config(
    config: dict,
    output_dir: Path,
    smell_code: str,
    sample: dict,
    index: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    project = safe_name(sample["project"].strip())
    target = safe_name(sample["target"].strip())

    config_path = output_dir / f"{index:04d}_{smell_code}_{project}_{target}.yml"

    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            config,
            file,
            sort_keys=False,
            allow_unicode=True,
        )

    return config_path


def run_main(config_path: Path) -> subprocess.CompletedProcess:
    command = [
        sys.executable,
        "main.py",
        "--config",
        str(config_path),
    ]

    return subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
    )


def main():
    args = parse_args()

    smell_code = args.smell
    csv_name = SMELL_TO_CSV[smell_code]

    base_config_path = Path(args.base_config)
    samples_path = Path(args.samples_dir) / csv_name
    generated_configs_dir = Path(args.generated_configs_dir) / smell_code

    base_config = load_base_config(base_config_path)
    samples = load_samples(samples_path)

    print(f"[INFO] Smell: {smell_code} ({SMELL_TO_FULL_NAME[smell_code]})")
    print(f"[INFO] Samples file: {samples_path}")
    print(f"[INFO] Total samples: {len(samples)}")
    print(f"[INFO] Generated configs dir: {generated_configs_dir}")

    successes = 0
    failures = 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_dir = Path("data/outputs/experiment_summaries")
    summary_dir.mkdir(parents=True, exist_ok=True)

    summary_path = summary_dir / f"{timestamp}_{smell_code}_summary.csv"
    summary_rows = []

    for index, sample in enumerate(samples, start=1):
        project = sample["project"].strip()
        target = sample["target"].strip()
        repo_path = sample["path"].strip()

        print()
        print(f"[RUN] {index}/{len(samples)}")
        print(f"      project: {project}")
        print(f"      repo:    {repo_path}")
        print(f"      target:  {target}")

        sample_config = build_sample_config(
            base_config=base_config,
            smell_code=smell_code,
            sample=sample,
            index=index,
        )

        generated_config_path = write_generated_config(
            config=sample_config,
            output_dir=generated_configs_dir,
            smell_code=smell_code,
            sample=sample,
            index=index,
        )

        result = run_main(generated_config_path)

        main_output = {}

        try:
            if result.stdout.strip():
                main_output = json.loads(result.stdout.strip())
        except json.JSONDecodeError:
            main_output = {}

        if result.returncode == 0:
            successes += 1
            print("[OK] Sample completed successfully.")
            if result.stdout.strip():
                print(result.stdout.strip())
        else:
            failures += 1
            print("[FAIL] Sample failed.")
            print(f"       Generated config: {generated_config_path}")

            if result.stdout.strip():
                print("[STDOUT]")
                print(result.stdout.strip())

            if result.stderr.strip():
                print("[STDERR]")
                print(result.stderr.strip())

            if args.stop_on_error:
                print("[STOP] Stopping because --stop-on-error was enabled.")
                break
        
        summary_rows.append({
            "index": index,
            "project": project,
            "repo_path": repo_path,
            "smell": smell_code,
            "target": target,
            "success": result.returncode == 0,
            "returncode": result.returncode,
            "generated_config": str(generated_config_path),
            "verdict_path": main_output.get("verdict_path", ""),
            "error": main_output.get("error", ""),
            "codex_returncode": main_output.get("codex_returncode", ""),
            "codex_stdout_path": main_output.get("codex_stdout_path", ""),
            "codex_stderr_path": main_output.get("codex_stderr_path", ""),
            "codex_prompt_path": main_output.get("codex_prompt_path", ""),
        })
    
    with summary_path.open("w", encoding="utf-8", newline="") as file:
        fieldnames = [
            "index",
            "project",
            "repo_path",
            "smell",
            "target",
            "success",
            "returncode",
            "generated_config",
            "verdict_path",
            "error",
            "codex_returncode",
            "codex_stdout_path",
            "codex_stderr_path",
            "codex_prompt_path",
        ]

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"  summary:   {summary_path}")

    print()
    print("[SUMMARY]")
    print(f"  smell:     {smell_code}")
    print(f"  total:     {len(samples)}")
    print(f"  successes: {successes}")
    print(f"  failures:  {failures}")

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())