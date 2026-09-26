import argparse
import math
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate metrics per smell comparing human labels and Codex predictions."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input CSV with columns: repository, smell, target, human_label, codex.",
    )

    parser.add_argument(
        "--output",
        default="data/outputs/human_codex_metrics_by_smell.csv",
        help="Output CSV file.",
    )

    return parser.parse_args()


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator != 0 else 0.0


def calculate_metrics(group: pd.DataFrame) -> dict:
    y_true = group["human_label"].astype(int)
    y_pred = group["codex"].astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    total = tp + tn + fp + fn

    accuracy = safe_div(tp + tn, total)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)

    mcc_denominator = math.sqrt(
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    )
    mcc = safe_div((tp * tn) - (fp * fn), mcc_denominator)

    po = accuracy

    p_human_1 = safe_div(tp + fn, total)
    p_human_0 = safe_div(tn + fp, total)
    p_codex_1 = safe_div(tp + fp, total)
    p_codex_0 = safe_div(tn + fn, total)

    pe = (p_human_1 * p_codex_1) + (p_human_0 * p_codex_0)
    kappa = safe_div(po - pe, 1 - pe)

    return {
        "total": total,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
        "kappa": kappa,
    }


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    required_columns = {"repository", "smell", "target", "human_label", "codex"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    results = []

    for smell, group in df.groupby("smell", sort=True):
        metrics = calculate_metrics(group)
        results.append(
            {
                "smell": smell,
                **metrics,
            }
        )

    result_df = pd.DataFrame(results)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)

    print(f"[OK] Metrics written to: {output_path}")
    print()
    print(result_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())