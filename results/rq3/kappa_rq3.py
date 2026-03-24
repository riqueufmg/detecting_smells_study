from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score

AUTHOR1_FILE = "RQ3_author1 - samples.csv"
AUTHOR2_FILE = "RQ3_author2 - samples.csv"

ID_COLUMN = "id"

CATEGORY_COLUMNS = [
    "size",
    "coupling",
    "stability",
    "cohesion",
    "structural complexity",
    "weak reasoning",
]

OUTPUT_RESULTS = "kappa_results.csv"
OUTPUT_CONTINGENCY = "kappa_contingency_tables.csv"

def normalize_binary(value):

    if pd.isna(value):
        return 0

    if isinstance(value, (int, float, np.integer, np.floating)):
        return 1 if value != 0 else 0

    text = str(value).strip().lower()

    positive_values = {
        "1", "x", "true", "yes", "y", "sim", "checked", "mark", "selected"
    }
    negative_values = {
        "0", "", "false", "no", "n", "não", "nao", "none"
    }

    if text in positive_values:
        return 1
    if text in negative_values:
        return 0

    return 1


def load_and_prepare(file_path, suffix):
    df = pd.read_csv(file_path)

    df = df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed:")], errors="ignore")

    required = [ID_COLUMN] + CATEGORY_COLUMNS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Arquivo {file_path} está sem colunas obrigatórias: {missing}")

    metadata_cols = [c for c in ["element", "detection", "smell", "llm", "approach", "justification"] if c in df.columns]
    keep_cols = [ID_COLUMN] + metadata_cols + CATEGORY_COLUMNS
    df = df[keep_cols].copy()

    for col in CATEGORY_COLUMNS:
        df[col] = df[col].apply(normalize_binary).astype(int)

    rename_map = {col: f"{col}_{suffix}" for col in CATEGORY_COLUMNS}
    for col in metadata_cols:
        rename_map[col] = f"{col}_{suffix}"

    df = df.rename(columns=rename_map)
    return df


def observed_agreement(y1, y2):
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)
    return float((y1 == y2).mean())


def contingency_counts(y1, y2):
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)

    a = int(((y1 == 1) & (y2 == 1)).sum())
    b = int(((y1 == 1) & (y2 == 0)).sum())
    c = int(((y1 == 0) & (y2 == 1)).sum())
    d = int(((y1 == 0) & (y2 == 0)).sum())

    return a, b, c, d


def interpret_kappa(kappa):
    if pd.isna(kappa):
        return "undefined"
    if kappa < 0:
        return "less than chance"
    if kappa < 0.21:
        return "slight"
    if kappa < 0.41:
        return "fair"
    if kappa < 0.61:
        return "moderate"
    if kappa < 0.81:
        return "substantial"
    return "almost perfect"

author1 = load_and_prepare(AUTHOR1_FILE, "A1")
author2 = load_and_prepare(AUTHOR2_FILE, "A2")

merged = pd.merge(author1, author2, on=ID_COLUMN, how="inner")

if len(merged) == 0:
    raise ValueError("Error.")

for meta in ["element", "smell", "llm", "approach"]:
    c1 = f"{meta}_A1"
    c2 = f"{meta}_A2"
    if c1 in merged.columns and c2 in merged.columns:
        mismatches = merged[merged[c1].astype(str) != merged[c2].astype(str)]
        if len(mismatches) > 0:
            print(f"[AVISO] Há {len(mismatches)} divergências na coluna de metadado '{meta}' entre os dois arquivos.")


results = []
contingency_rows = []

for category in CATEGORY_COLUMNS:
    col_a1 = f"{category}_A1"
    col_a2 = f"{category}_A2"

    y1 = merged[col_a1]
    y2 = merged[col_a2]

    kappa = cohen_kappa_score(y1, y2)

    po = observed_agreement(y1, y2)

    prevalence_a1 = float(y1.mean())
    prevalence_a2 = float(y2.mean())

    both_1, a1_only, a2_only, both_0 = contingency_counts(y1, y2)

    results.append({
        "category": category,
        "n_items": len(merged),
        "kappa": round(kappa, 4),
        "agreement_observed": round(po, 4),
        "author1_positive_rate": round(prevalence_a1, 4),
        "author2_positive_rate": round(prevalence_a2, 4),
        "interpretation": interpret_kappa(kappa),
    })

    contingency_rows.append({
        "category": category,
        "both_marked_1": both_1,
        "author1_only": a1_only,
        "author2_only": a2_only,
        "both_marked_0": both_0,
    })

results_df = pd.DataFrame(results).sort_values(by="category")
contingency_df = pd.DataFrame(contingency_rows).sort_values(by="category")

mean_kappa = results_df["kappa"].mean()

results_df.to_csv(OUTPUT_RESULTS, index=False, encoding="utf-8-sig")
#contingency_df.to_csv(OUTPUT_CONTINGENCY, index=False, encoding="utf-8-sig")