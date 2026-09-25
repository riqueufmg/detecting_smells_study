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
    if isinstance(value, bool):
        return value

    if isinstance(value, int):
        if value == 1:
            return True
        if value == 0:
            return False

    if isinstance(value, str):
        value = value.strip().lower()

        if value in {"true", "1", "yes"}:
            return True

        if value in {"false", "0", "no"}:
            return False

    raise ValueError(f"Cannot convert to boolean: {value!r}")


def target_to_filename(target):
    """
    Raw-code outputs use '_' instead of '.'.

    Example:
        com.github.javaparser.ast.Node
        ->
        com_github_javaparser_ast_Node.txt
    """
    return f"{target.replace('.', '_')}.txt"


def extract_json_object(text):
    cleaned = re.sub(
        r"```(?:json)?",
        "",
        text,
        flags=re.IGNORECASE
    )

    cleaned = cleaned.replace("```", "").strip()

    cleaned = re.sub(
        r"^\s*json\s*",
        "",
        cleaned,
        flags=re.IGNORECASE
    )

    try:
        data = json.loads(cleaned)

        if isinstance(data, dict):
            return data

    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()

    for match in re.finditer(r"\{", cleaned):
        try:
            data, _ = decoder.raw_decode(
                cleaned[match.start():]
            )

            if isinstance(data, dict):
                return data

        except json.JSONDecodeError:
            continue

    return None


def extract_field_with_regex(text, field):
    if field == "detection":
        pattern = (
            r'["\']?detection["\']?\s*:\s*'
            r'(true|false|1|0)'
        )

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        if match:
            return parse_bool(match.group(1))

        return None

    pattern = (
        rf'["\']?{re.escape(field)}["\']?'
        rf'\s*:\s*["\']([^"\']+)["\']'
    )

    match = re.search(
        pattern,
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    if match:
        return match.group(1).strip()

    return None


def extract_justification_with_regex(text):
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

            try:
                return json.loads(f'"{value}"')
            except Exception:
                return value

    return None


def parse_llm_output(file_path, target_field):
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
        target = data.get(target_field)

        if target is None:
            target = data.get("target")

        detection = data.get("detection")
        justification = data.get("justification")

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
            f"Could not find '{target_field}'"
        )

    if detection is None:
        raise ValueError(
            "Could not find 'detection'"
        )

    if justification is None:
        raise ValueError(
            "Could not find 'justification'"
        )

    return {
        "target": target.strip(),
        "detection": parse_bool(detection),
        "justification": justification.strip(),
    }

def load_ground_truth(sample_file):
    rows = []

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
                f"Missing required columns in {sample_file}"
            )

        for row in reader:
            rows.append({
                "repository": row["repository"].strip(),
                "target": row["target"].strip(),
                "human_label": parse_bool(
                    row["human_label"]
                ),
            })

    return rows


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
        if total else 0.0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    # Matthews Correlation Coefficient (MCC)
    mcc_denominator = (
        (tp + fp)
        * (tp + fn)
        * (tn + fp)
        * (tn + fn)
    ) ** 0.5

    mcc = (
        ((tp * tn) - (fp * fn)) / mcc_denominator
        if mcc_denominator
        else 0.0
    )

    # Cohen's Kappa
    if total:
        observed = accuracy

        actual_positive = tp + fn
        actual_negative = tn + fp

        predicted_positive = tp + fp
        predicted_negative = tn + fn

        expected = (
            actual_positive * predicted_positive
            + actual_negative * predicted_negative
        ) / (total * total)

        if expected != 1:
            kappa = (
                observed - expected
            ) / (
                1 - expected
            )
        else:
            kappa = 1.0

    else:
        kappa = 0.0

    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mcc": round(mcc, 4),
        "cohen_kappa": round(kappa, 4),
    }

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--smell",
        required=True,
        choices=SMELL_CONFIG.keys(),
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=MODELS,
    )

    args = parser.parse_args()

    smell = args.smell
    model = args.model

    config = SMELL_CONFIG[smell]

    sample_file = (
        Path("data/sample")
        / config["sample_file"]
    )

    if not sample_file.exists():
        raise FileNotFoundError(
            f"Sample file not found: {sample_file}"
        )

    input_dir = Path(
        "raw_code",
        "data",
        "processed",
        "llm_outputs",
        smell,
        model
    )

    if not input_dir.exists():
        raise FileNotFoundError(
            f"LLM output directory not found: {input_dir}"
        )

    output_dir = Path(
        "data",
        "results",
        smell,
        model,
        "raw_code"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    ground_truth = load_ground_truth(
        sample_file
    )

    records = []
    missing = []
    errors = []

    for sample in ground_truth:

        repository = sample["repository"]
        expected_target = sample["target"]

        filename = target_to_filename(
            expected_target
        )

        output_file = input_dir / filename

        if not output_file.exists():
            missing.append({
                "repository": repository,
                "target": expected_target,
                "expected_file": str(output_file),
            })

            print(
                f"[MISSING] {repository} | "
                f"{expected_target}"
            )

            continue

        try:
            parsed = parse_llm_output(
                output_file,
                config["target_field"]
            )

        except Exception as error:
            errors.append({
                "repository": repository,
                "target": expected_target,
                "file": str(output_file),
                "error": str(error),
            })

            print(
                f"[ERROR] {output_file}: {error}"
            )

            continue

        #
        # Important consistency check:
        # target in output should match target in ground truth.
        #
        if parsed["target"] != expected_target:
            print(
                f"[WARNING] Target mismatch:\n"
                f"  CSV: {expected_target}\n"
                f"  LLM: {parsed['target']}"
            )

        records.append({
            "repository": repository,
            "target": expected_target,
            "detection": parsed["detection"],
            "justification": parsed["justification"],
            "human_label": sample["human_label"],
        })

    #
    # Consolidated results
    #
    results_file = output_dir / "results.json"

    public_results = [
        {
            "repository": r["repository"],
            "target": r["target"],
            "detection": r["detection"],
            "justification": r["justification"],
        }
        for r in records
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
    # Metrics
    #
    metrics = calculate_metrics(records)

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
    print(f"Valid results: {len(records)}")
    print(f"Missing outputs: {len(missing)}")
    print(f"Parsing errors: {len(errors)}")
    print()

    for metric, value in metrics.items():
        print(f"{metric}: {value}")

    print()
    print(f"Results: {results_file}")
    print(f"Metrics: {metrics_file}")

    if missing:
        missing_file = output_dir / "missing_results.json"

        with missing_file.open(
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                missing,
                f,
                indent=2,
                ensure_ascii=False
            )

        print(f"Missing: {missing_file}")

    if errors:
        errors_file = output_dir / "parsing_errors.json"

        with errors_file.open(
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                errors,
                f,
                indent=2,
                ensure_ascii=False
            )

        print(f"Parsing errors: {errors_file}")


if __name__ == "__main__":
    main()