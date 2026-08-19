from pathlib import Path
import argparse
import csv
import shutil


SMELL_CONFIG = {
    "IM": {
        "csv": "IM.csv",
        "source_dir": "insufficient_modularization",
        "target_type": "class",
    },
    "GC": {
        "csv": "GC.csv",
        "source_dir": "god_component",
        "target_type": "package",
    },
    "UD": {
        "csv": "UD.csv",
        "source_dir": "unstable_dependency",
        "target_type": "package",
    },
    "HM": {
        "csv": "HM.csv",
        "source_dir": "hublike_modularization",
        "target_type": "class",
    },
}


def target_to_filename(target: str, target_type: str) -> str:
    return f"{target.replace('.', '_')}.txt"


def main():
    parser = argparse.ArgumentParser(
        description="Filter generated prompts according to smell samples."
    )

    parser.add_argument(
        "--smell",
        required=True,
        choices=SMELL_CONFIG.keys(),
        help="Smell to process: IM, GC, UD, or HM",
    )

    args = parser.parse_args()

    smell = args.smell
    config = SMELL_CONFIG[smell]

    sample_file = Path("data/sample") / config["csv"]

    source_dir = (
        Path("data/processed/prompts/smell_detection")
        / config["source_dir"]
    )

    output_dir = Path("data/processed/prompts") / smell

    if not sample_file.exists():
        raise FileNotFoundError(
            f"Sample file not found: {sample_file}"
        )

    if not source_dir.exists():
        raise FileNotFoundError(
            f"Prompt directory not found: {source_dir}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    missing = []

    with sample_file.open(
        "r",
        encoding="utf-8",
        newline=""
    ) as csv_file:

        reader = csv.DictReader(csv_file)

        required_columns = {"repository", "target"}

        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError(
                f"CSV must contain columns: "
                f"{sorted(required_columns)}. "
                f"Found: {reader.fieldnames}"
            )

        for row in reader:
            repository = row["repository"].strip()
            target = row["target"].strip()

            filename = target_to_filename(
                target,
                config["target_type"]
            )

            source_file = (
                source_dir
                / repository
                / filename
            )

            destination_dir = (
                output_dir
                / repository
            )

            destination_file = (
                destination_dir
                / filename
            )

            if not source_file.exists():
                missing.append(source_file)
                continue

            destination_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            shutil.copy2(
                source_file,
                destination_file
            )

            copied += 1

    print()
    print(f"Smell: {smell}")
    print(f"Sample file: {sample_file}")
    print(f"Copied prompts: {copied}")
    print(f"Missing prompts: {len(missing)}")

    if missing:
        print("\nMissing files:")

        for path in missing:
            print(f"  {path}")


if __name__ == "__main__":
    main()