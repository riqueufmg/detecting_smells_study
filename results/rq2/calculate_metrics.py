import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    cohen_kappa_score
)

FILE_PATH = "metrics_predictions.csv"
SEP = ";"
HUMAN_COL = "human_label"
LLM_COLS = ["gpt", "deepseek", "qwen"]
GROUP_COL = "architecture_smell"

df = pd.read_csv(FILE_PATH, sep=";")

cols_to_numeric = [HUMAN_COL] + LLM_COLS
for col in cols_to_numeric:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=cols_to_numeric + [GROUP_COL]).copy()

for col in cols_to_numeric:
    df[col] = df[col].astype(int)

def compute_metrics(y_true, y_pred):
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "support": len(y_true)
    }

overall_results = []
for llm in LLM_COLS:
    metrics = compute_metrics(df[HUMAN_COL], df[llm])
    metrics["model"] = llm
    metrics["architecture_smell"] = "ALL"
    overall_results.append(metrics)

by_smell_results = []

for smell, group in df.groupby(GROUP_COL):
    for llm in LLM_COLS:
        metrics = compute_metrics(group[HUMAN_COL], group[llm])
        metrics["model"] = llm
        metrics["architecture_smell"] = smell
        by_smell_results.append(metrics)

results_df = pd.DataFrame(overall_results + by_smell_results)

results_df = results_df[
    [
        "architecture_smell",
        "model",
        "precision",
        "recall",
        "f1",
        "mcc",
        "cohen_kappa",
        "support"
    ]
]

results_df = results_df.sort_values(
    by=["architecture_smell", "model"]
).reset_index(drop=True)

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

results_df.to_csv("metrics_rq2.csv", index=False, float_format="%.2f")
print("\nSaved file: metrics_rq2.csv")

formatted_df = results_df.copy()
metric_cols = ["precision", "recall", "f1", "mcc", "cohen_kappa"]
formatted_df[metric_cols] = formatted_df[metric_cols].round(2)