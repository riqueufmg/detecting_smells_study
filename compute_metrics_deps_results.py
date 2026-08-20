from pathlib import Path
import argparse
import csv
import json
import re


SMELL_CONFIG = {
    "insufficient_modularization": {
        "sample_file": "IM.csv",
        "target_field": "class",
    },
    "hublike_modularization": {
        "sample_file": "HM.csv",
        "target_field": "class",
    },
    "god_component": {
        "sample_file": "GC.csv",
        "target_field": "package",
    },
    "unstable_dependency": {
        "sample_file": "UD.csv",
        "target_field": "package",
    },
}

MODELS = {
    "deepseek",
    "gpt",
    "kimi-k3",
    "qwen",
}


def parse_bool(value):
    """
    Convert different representations to bool.
    """
    if isinstance(value, bool):
        return value

    if isinstance(value, int):
        if value == 1:
            return True
        if value == 0:
            return False

    if isinstance(value, str):
        normalized = value.strip().lower()

        if normalized in {"true", "1", "yes"}:
            return True

        if normalized in {"false", "0", "no"}:
            return False

    raise ValueError(f"Cannot convert to boolean: {value!r}")


def extract_json_object(text):
    """
    Try to extract a JSON object from arbitrary LLM output.

    Supports, for example:

        ```json
        {...}
        ```

        json
        {...}

        Some explanation...
        {...}
        More text...
    """

    # Remove Markdown fences.
    cleaned = re.sub(
        r"```(?:json)?",
        "",
        text,
        flags=re.IGNORECASE
    )
    cleaned = cleaned.replace("```", "").strip()

    # Remove a standalone leading "json"
    cleaned = re.sub(
        r"^\s*json\s*",
        "",
        cleaned,
        flags=re.IGNORECASE
    )

    # First try the entire content.
    try:
        data = json.loads(cleaned)

        if isinstance(data, dict):
            return data

    except json.JSONDecodeError:
        pass

    # Then search for every possible JSON object.
    decoder = json.JSONDecoder()

    for match in re.finditer(r"\{", cleaned):
        start = match.start()

        try:
            data, _ = decoder.raw_decode(cleaned[start:])

            if isinstance(data, dict):
                return data

        except json.JSONDecodeError:
            continue

    return None


def extract_field_with_regex(text, field):
    """
    Fallback for malformed JSON.

    Examples:
        "detection": true
        "package": "org.example.foo"
    """

    if field == "detection":
        pattern = r'["\']?detection["\']?\s*:\s*(true|false|1|0)'
        match = re.search(pattern, text, flags=re.IGNORECASE)

        if match:
            return parse_bool(match.group(1))

        return None

    pattern = rf'["\']?{re.escape(field)}["\']?\s*:\s*["\']([^"\']+)["\']'

    match = re.search(
        pattern,
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    if match:
        return match.group(1).strip()

    return None


def extract_justification_with_regex(text):
    """
    Fallback extraction for justification in malformed JSON.
    """

    patterns = [
        r'"justification"\s*:\s*"((?:\\.|[^"\\])*)"',
        r"'justification'\s*:\s*'((?:\\.|[^'\\])*)'",
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE | re.DOTALL
        )

        if match:
            value = match.group(1)

            # Decode common escaped characters if possible.
            try:
                return json.loads(f'"{value}"')
            except Exception:
                return value

    return None


def parse_llm_output(file_path, target_field):
    """
    Parse one LLM output.

    Returns:
        target
        detection
        justification
    """

    text = file_path.read_text(
        encoding="utf-8",
        errors="replace"
    ).strip()

    if not text:
        raise ValueError("Empty output file")

    data = extract_json_object(text)

    target = None
    detection = None
    justification = None

    if data:
        # Normally class smells have "class"
        # and package smells have "package".
        target = data.get(target_field)

        # Also tolerate the generic field "target".
        if target is None:
            target = data.get("target")

        detection = data.get("detection")
        justification = data.get("justification")

    # Fall back to regex if JSON parsing failed or fields are missing.
    if target is None:
        target = extract_field_with_regex(
            text,
            target_field
        )

    if detection is None:
        detection = extract_field_with_regex(
            text,
            "detection"
        )

    if justification is None:
        justification = extract_justification_with_regex(
            text
        )

    if target is None:
        raise ValueError(
            f"Could not find '{target_field}' in output"
        )

    if detection is None:
        raise ValueError(
            "Could not find 'detection' in output"
        )

    if justification is None:
        raise ValueError(
            "Could not find 'justification' in output"
        )

    return {
        "target": target.strip(),
        "detection": parse_bool(detection),
        "justification": justification.strip(),
    }


def find_sample_file(filename):
    """
    Prefer the sample directory at project root.

    Also support the older metrics_deps/data/sample layout.
    """

    candidates = [
        Path("data/sample") / filename,
        Path("metrics_deps/data/sample") / filename,
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Sample file not found. Tried:\n"
        + "\n".join(f"  - {path}" for path in candidates)
    )


def load_ground_truth(sample_file):
    """
    Load repository + target -> human_label.
    """

    ground_truth = {}

    with sample_file.open(
        "r",
        encoding="utf-8",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        required = {
            "repository",
            "target",
            "human_label",
        }

        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"{sample_file} must contain columns "
                f"{sorted(required)}. "
                f"Found: {reader.fieldnames}"
            )

        for row in reader:
            repository = row["repository"].strip()
            target = row["target"].strip()

            ground_truth[(repository, target)] = parse_bool(
                row["human_label"]
            )

    return ground_truth


def calculate_metrics(records):
    tp = tn = fp = fn = 0

    for record in records:
        actual = record["human_label"]
        predicted = record["detection"]

        if actual and predicted:
            tp += 1

        elif not actual and not predicted:
            tn += 1

        elif not actual and predicted:
            fp += 1

        elif actual and not predicted:
            fn += 1

    total = tp + tn + fp + fn

    accuracy = (
        (tp + tn) / total
        if total
        else 0.0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp)
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn)
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    # Cohen's Kappa
    if total:
        observed_agreement = accuracy

        actual_positive = tp + fn
        actual_negative = tn + fp

        predicted_positive = tp + fp
        predicted_negative = tn + fn

        expected_agreement = (
            (actual_positive * predicted_positive)
            + (actual_negative * predicted_negative)
        ) / (total * total)

        if expected_agreement != 1:
            cohen_kappa = (
                observed_agreement - expected_agreement
            ) / (
                1 - expected_agreement
            )
        else:
            cohen_kappa = 1.0
    else:
        cohen_kappa = 0.0

    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "cohen_kappa": round(cohen_kappa, 4),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Consolidate metrics+dependencies smell detection "
            "results and compute classification metrics."
        )
    )

    parser.add_argument(
        "--smell",
        required=True,
        choices=SMELL_CONFIG.keys(),
        help=(
            "Target smell: insufficient_modularization, "
            "hublike_modularization, god_component, "
            "or unstable_dependency"
        ),
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=MODELS,
        help="Model: deepseek, gpt, kimi-k3, or qwen",
    )

    args = parser.parse_args()

    smell = args.smell
    model = args.model

    config = SMELL_CONFIG[smell]

    sample_file = find_sample_file(
        config["sample_file"]
    )

    ground_truth = load_ground_truth(
        sample_file
    )

    input_base = Path(
        "metrics_deps",
        "data",
        "processed",
        "llm_outputs"
    )

    output_dir = Path(
        "data",
        "results",
        smell,
        model,
        "metrics_deps"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    results = []
    errors = []
    unmatched = []

    # The first directory level is the repository.
    for repository_dir in sorted(input_base.iterdir()):

        if not repository_dir.is_dir():
            continue

        repository = repository_dir.name

        model_dir = (
            repository_dir
            / smell
            / model
        )

        if not model_dir.exists():
            continue

        for output_file in sorted(model_dir.glob("*.txt")):

            try:
                parsed = parse_llm_output(
                    output_file,
                    config["target_field"]
                )

            except Exception as error:
                errors.append({
                    "repository": repository,
                    "file": str(output_file),
                    "error": str(error),
                })

                print(
                    f"[ERROR] {output_file}: {error}"
                )

                continue

            target = parsed["target"]

            key = (repository, target)

            if key not in ground_truth:
                unmatched.append({
                    "repository": repository,
                    "target": target,
                    "file": str(output_file),
                })

                print(
                    f"[WARNING] No human label for: "
                    f"{repository} | {target}"
                )

                continue

            results.append({
                "repository": repository,
                "target": target,
                "detection": parsed["detection"],
                "justification": parsed["justification"],
                "human_label": ground_truth[key],
            })

    #
    # Save requested consolidated result file.
    #
    results_file = output_dir / "results.json"

    public_results = [
        {
            "repository": record["repository"],
            "target": record["target"],
            "detection": record["detection"],
            "justification": record["justification"],
        }
        for record in results
    ]

    with results_file.open(
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            public_results,
            f,
            indent=2,
            ensure_ascii=False
        )

    #
    # Calculate and save aggregate metrics.
    #
    metrics = calculate_metrics(results)

    metrics_file = output_dir / "metrics.json"

    with metrics_file.open(
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            metrics,
            f,
            indent=2,
            ensure_ascii=False
        )

    print()
    print("=" * 60)
    print(f"Smell: {smell}")
    print(f"Model: {model}")
    print(f"Sample: {sample_file}")
    print(f"Valid results: {len(results)}")
    print(f"Parsing errors: {len(errors)}")
    print(f"Unmatched results: {len(unmatched)}")
    print()

    print("Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value}")

    print()
    print(f"Results: {results_file}")
    print(f"Metrics: {metrics_file}")

    if errors:
        error_file = output_dir / "parsing_errors.json"

        with error_file.open(
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                errors,
                f,
                indent=2,
                ensure_ascii=False
            )

        print(f"Parsing errors: {error_file}")

    if unmatched:
        unmatched_file = output_dir / "unmatched_results.json"

        with unmatched_file.open(
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                unmatched,
                f,
                indent=2,
                ensure_ascii=False
            )

        print(f"Unmatched results: {unmatched_file}")


if __name__ == "__main__":
    main()