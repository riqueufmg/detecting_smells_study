import pandas as pd
import numpy as np
import csv

INPUT_FILES = ["metrics_results", "code_results"]

GROUND_TRUTH_COL = "designite"
SMELL_COL = "smell"
LLM_COLS = ["gpt", "deepseek", "qwen"]

def sniff_delimiter(path: str) -> str:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
    try:
        return csv.Sniffer().sniff(sample, delimiters=";,").delimiter
    except Exception:
        return ";"

def to_binary_series(s):
    if pd.api.types.is_numeric_dtype(s):
        return (s.fillna(0).astype(float) > 0).astype(int)
    return (s.astype(str).str.strip().isin(["1", "True", "true"])).astype(int)

def compute_confusion(gt, pred):
    tp = ((gt == 1) & (pred == 1)).sum()
    tn = ((gt == 0) & (pred == 0)).sum()
    fp = ((gt == 0) & (pred == 1)).sum()
    fn = ((gt == 1) & (pred == 0)).sum()
    return tp, tn, fp, fn

def compute_metrics(tp, tn, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    #specificity = tn / (tn + fp) if (tn + fp) else 0.0
    #balanced_acc = (recall + specificity) / 2

    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / denom) if denom else 0.0

    return {
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "Precision": round(precision, 2),
        "Recall": round(recall, 2),
        "F1": round(f1, 2),
        #"Specificity": round(specificity, 4),
        #"Balanced_Accuracy": round(balanced_acc, 4),
        "MCC": round(mcc, 2),
        #"Support_Positive": int(tp + fn),
        #"Support_Negative": int(tn + fp),
    }

def main():
    
    for file in INPUT_FILES:

        file = f"{file}.csv"

        delim = sniff_delimiter(file)
        df = pd.read_csv(file, sep=delim, encoding="utf-8-sig")

        df[GROUND_TRUTH_COL] = to_binary_series(df[GROUND_TRUTH_COL])
        for llm in LLM_COLS:
            df[llm] = to_binary_series(df[llm])

        results = []

        for smell in sorted(df[SMELL_COL].unique()):
            df_smell = df[df[SMELL_COL] == smell]

            print(f"\n--- Smell: {smell} ---")

            for llm in LLM_COLS:
                tp, tn, fp, fn = compute_confusion(
                    df_smell[GROUND_TRUTH_COL],
                    df_smell[llm]
                )

                metrics = compute_metrics(tp, tn, fp, fn)
                metrics["Smell"] = smell
                metrics["LLM"] = llm

                results.append(metrics)

                print(f"\nLLM: {llm}")
                for k, v in metrics.items():
                    if k not in ["Smell", "LLM"]:
                        print(f"{k}: {v}")

        results_df = pd.DataFrame(results)
        results_df.to_csv(f"{file}_per_smell.csv", index=False)

        print(f"\nResults saved to {file}_per_smell.csv")

if __name__ == "__main__":
    main()