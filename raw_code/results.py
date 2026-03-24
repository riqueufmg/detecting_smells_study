import csv
import json
from pathlib import Path

LLM = "qwen"
BASE_DIR = Path("data/processed")
CANDIDATES_DIR = BASE_DIR / "candidates_sampled"
LLM_OUTPUTS_DIR = BASE_DIR / "llm_outputs"
OUTPUT_DIR = BASE_DIR / "results" / LLM

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SMELLS = {
    "god_component": "god_component_sample.csv",
    "unstable_dependency": "unstable_dependency_sample.csv",
    "insufficient_modularization": "insufficient_modularization_sample.csv",
    "hublike_modularization": "hublike_modularization_sample.csv",
}


def load_detection(llm_file: Path) -> bool:
    content = llm_file.read_text(encoding="utf-8").strip()

    if not content:
        return False

    try:
        data = json.loads(content)
        return bool(data.get("detection", False))
    except json.JSONDecodeError:
        pass

    start = content.find("{")
    end = content.rfind("}")

    if start == -1 or end == -1 or end <= start:
        return False

    try:
        data = json.loads(content[start:end + 1])
        return bool(data.get("detection", False))
    except json.JSONDecodeError:
        return False


def compute_metrics(tp, tn, fp, fn):
    total = tp + tn + fp + fn

    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    # Cohen's Kappa
    if total:
        p_observed = (tp + tn) / total

        pred_positive = (tp + fp) / total
        pred_negative = (tn + fn) / total
        gold_positive = (tp + fn) / total
        gold_negative = (tn + fp) / total

        p_expected = (pred_positive * gold_positive) + (pred_negative * gold_negative)

        if (1 - p_expected) == 0:
            cohen_kappa = 0.0
        else:
            cohen_kappa = (p_observed - p_expected) / (1 - p_expected)
    else:
        cohen_kappa = 0.0

    return {
        "accuracy": round(accuracy, 3),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "cohen_kappa": round(cohen_kappa, 3),
    }


def safe_int(value, default=0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def fqn_from_prompt_filename(prompt_filename: str) -> str:
    # Fallback: org_apache_commons_lang3_math.txt -> org.apache.commons.lang3.math
    name = Path(prompt_filename).stem
    name = name.strip()
    if name == "(default package)":
        return "(default package)"
    return name.replace("_", ".")


def first_nonempty(row: dict, keys: list[str]) -> str:
    for k in keys:
        v = (row.get(k) or "").strip()
        if v:
            return v
    return ""


def build_entity_id(row: dict, smell_name: str, prompt_filename: str) -> str:
    """
    For package-level smells (god_component, unstable_dependency): return package id.
    For class-level smells (insufficient_modularization, hublike_modularization): return package.class if possible.
    Fallback to prompt filename decoding.
    """
    direct = first_nonempty(row, ["fqn", "entity", "qualified_name", "qualifiedName"])
    if direct:
        return direct

    if smell_name in {"god_component", "unstable_dependency"}:
        pkg = first_nonempty(row, ["package", "package_name", "pkg", "package_fqn"])
        if pkg:
            return pkg
        return fqn_from_prompt_filename(prompt_filename)

    pkg = first_nonempty(row, ["package", "package_name", "pkg", "package_fqn"])
    cls = first_nonempty(row, ["class", "class_name", "type", "classname", "className"])

    if pkg and cls:
        return f"{pkg}.{cls}"
    if cls:
        return cls
    if pkg:
        return pkg

    return fqn_from_prompt_filename(prompt_filename)


def process_smell(smell_name: str, csv_name: str):
    csv_path = CANDIDATES_DIR / csv_name
    llm_dir = LLM_OUTPUTS_DIR / smell_name / LLM

    tp = tn = fp = fn = 0
    samples = []

    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            label = safe_int(row.get("label"))
            prompt_file = Path(row["prompt_file"])
            llm_file = llm_dir / prompt_file.name

            detection = load_detection(llm_file) if llm_file.exists() else False
            print(f"Processed: {llm_file}")

            prediction = 1 if detection else 0

            if label == 1 and prediction == 1:
                tp += 1
                cls = "tp"
            elif label == 0 and prediction == 0:
                tn += 1
                cls = "tn"
            elif label == 0 and prediction == 1:
                fp += 1
                cls = "fp"
            else:
                fn += 1
                cls = "fn"

            context_size = safe_int(row.get("context_size"), default=-1)
            entity_id = build_entity_id(row, smell_name, prompt_file.name)

            samples.append(
                {
                    "smell": smell_name,
                    "classification": cls,
                    "context_size": context_size,
                    "fqn": entity_id,
                    "label": label,
                    "prediction": prediction,
                    "prompt_file": prompt_file.name,
                }
            )

    metrics = compute_metrics(tp, tn, fp, fn)

    result = {
        "confusion_matrix": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
        "metrics": metrics,
    }

    return result, samples


def write_samples_csv(path: Path, rows: list):
    fieldnames = [
        "smell",
        "classification",
        "context_size",
        "fqn",
        "label",
        "prediction",
        "prompt_file",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main():
    all_results = {}
    all_samples = []

    for smell, csv_file in SMELLS.items():
        print(f"Processing smell: {smell}")
        result, samples = process_smell(smell, csv_file)

        all_results[smell] = result
        all_samples.extend(samples)

        output_file = OUTPUT_DIR / f"{smell}_results.json"
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=4)

        details_csv = OUTPUT_DIR / f"{smell}_samples_details.csv"
        write_samples_csv(details_csv, samples)

    aggregated_file = OUTPUT_DIR / "all_smells_results.json"
    with aggregated_file.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)

    all_details_csv = OUTPUT_DIR / "all_samples_details.csv"
    write_samples_csv(all_details_csv, all_samples)

    print("\nDone. Results saved in:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()