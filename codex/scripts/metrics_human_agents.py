import argparse
import math
from pathlib import Path

import pandas as pd


DEFAULT_MODEL_COLUMNS = ["codex", "gtp", "deepseek", "qwen"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Calculate metrics per smell and model comparing human labels "
            "against multiple agent/LLM predictions."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Input CSV with columns: repository, smell, target, human_label, "
            "codex, gtp, deepseek, qwen."
        ),
    )

    parser.add_argument(
        "--output",
        default="data/outputs/human_agents_metrics_by_smell_model.csv",
        help="Output CSV file.",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODEL_COLUMNS,
        help="Prediction columns to evaluate. Default: codex gtp deepseek qwen.",
    )

    return parser.parse_args()


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator != 0 else 0.0


def normalize_binary_series(series: pd.Series, column_name: str) -> pd.Series:
    values = series.astype(str).str.strip().str.lower()

    mapping = {
        "0": 0,
        "0.0": 0,
        "false": 0,
        "no": 0,
        "1": 1,
        "1.0": 1,
        "true": 1,
        "yes": 1,
    }

    normalized = values.map(mapping)

    if normalized.isna().any():
        invalid_values = sorted(series[normalized.isna()].astype(str).unique())
        raise ValueError(
            f"Column '{column_name}' has invalid binary values: {invalid_values}"
        )

    return normalized.astype(int)


def calculate_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
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
    p_pred_1 = safe_div(tp + fp, total)
    p_pred_0 = safe_div(tn + fn, total)

    pe = (p_human_1 * p_pred_1) + (p_human_0 * p_pred_0)
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


def pretty_model_name(model_column: str) -> str:
    mapping = {
        "codex": "Codex",
        "gtp": "GPT",
        "gpt": "GPT",
        "deepseek": "DeepSeek",
        "qwen": "Qwen",
    }

    return mapping.get(model_column, model_column)


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    required_columns = {"repository", "smell", "target", "human_label"}
    required_columns.update(args.models)

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["human_label"] = normalize_binary_series(df["human_label"], "human_label")

    for model_column in args.models:
        df[model_column] = normalize_binary_series(df[model_column], model_column)

    results = []

    for smell, group in df.groupby("smell", sort=True):
        y_true = group["human_label"]

        for model_column in args.models:
            y_pred = group[model_column]
            metrics = calculate_metrics(y_true, y_pred)

            results.append(
                {
                    "smell": smell,
                    "model": pretty_model_name(model_column),
                    "model_column": model_column,
                    **metrics,
                }
            )

    result_df = pd.DataFrame(results)

    metric_columns = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "mcc",
        "kappa",
    ]

    result_df[metric_columns] = result_df[metric_columns].round(2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False, float_format="%.2f")

    print(f"[OK] Metrics written to: {output_path}")
    print()

    print(
        result_df[
            [
                "smell",
                "model",
                "total",
                "tp",
                "tn",
                "fp",
                "fn",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "mcc",
                "kappa",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())