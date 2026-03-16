import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import math

# ==========================
# CONFIG
# ==========================
INPUT_FILE_1 = "metrics_results.csv"
INPUT_FILE_2 = "code_results.csv"

LABEL_1 = "metrics"
LABEL_2 = "code"

OUTPUT_FILE_PNG = "heatmap_compare_approaches.png"
OUTPUT_FILE_PDF = "heatmap_compare_approaches.pdf"

SMELL_COL = "smell"
GROUND_TRUTH_COL = "designite"
LLM_COLS = ["gpt", "deepseek", "qwen"]

SMELL_ABBR = {
    "God Component": "GC",
    "Unstable Dependency": "UD",
    "Insufficient Modularization": "IM",
    "Hub-like Modularization": "HM",
}

# ==========================
# HELPERS
# ==========================
def sniff_delimiter(path: str) -> str:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
    try:
        return csv.Sniffer().sniff(sample, delimiters=";,").delimiter
    except Exception:
        return ";"

def to_binary_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return (s.fillna(0).astype(float) > 0).astype(int)
    v = s.fillna("").astype(str).str.strip().str.lower()
    return v.isin(["1", "true", "yes", "y"]).astype(int)

def compute_confusion(gt, pred):
    tp = int(((gt == 1) & (pred == 1)).sum())
    tn = int(((gt == 0) & (pred == 0)).sum())
    fp = int(((gt == 0) & (pred == 1)).sum())
    fn = int(((gt == 1) & (pred == 0)).sum())
    return tp, tn, fp, fn

def compute_f1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

def compute_mcc(tp, tn, fp, fn):
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denom == 0:
        return 0.0
    return (tp * tn - fp * fn) / denom

def abbreviate_smell(name: str) -> str:
    name = str(name).strip()
    return SMELL_ABBR.get(name, name)

def load_and_compute(input_file: str):
    delim = sniff_delimiter(input_file)
    df = pd.read_csv(input_file, sep=delim, encoding="utf-8-sig")
    df.columns = [c.strip().lower() for c in df.columns]

    # normaliza colunas principais
    df[SMELL_COL] = df[SMELL_COL].astype(str).str.strip()
    df[GROUND_TRUTH_COL] = to_binary_series(df[GROUND_TRUTH_COL])

    for llm in LLM_COLS:
        df[llm] = to_binary_series(df[llm])

    smells_full = sorted(df[SMELL_COL].unique())

    f1_dict = {}
    mcc_dict = {}

    for smell in smells_full:
        sub = df[df[SMELL_COL] == smell]
        gt = sub[GROUND_TRUTH_COL]

        f1_dict[smell] = {}
        mcc_dict[smell] = {}

        for llm in LLM_COLS:
            tp, tn, fp, fn = compute_confusion(gt, sub[llm])
            f1_dict[smell][llm] = compute_f1(tp, fp, fn)
            mcc_dict[smell][llm] = compute_mcc(tp, tn, fp, fn)

    return smells_full, f1_dict, mcc_dict

# ==========================
# LOAD BOTH FILES
# ==========================
smells_1, f1_1, mcc_1 = load_and_compute(INPUT_FILE_1)
smells_2, f1_2, mcc_2 = load_and_compute(INPUT_FILE_2)

# união dos smells dos dois arquivos
smells_full = sorted(set(smells_1).union(set(smells_2)))
smells = [abbreviate_smell(s) for s in smells_full]

# colunas intercaladas para facilitar comparação visual
x_labels = []
for llm in LLM_COLS:
    x_labels.append(f"{llm}\n({LABEL_2})")
    x_labels.append(f"{llm}\n({LABEL_1})")

f1_matrix = np.full((len(smells_full), len(x_labels)), np.nan)
mcc_matrix = np.full((len(smells_full), len(x_labels)), np.nan)

for i, smell in enumerate(smells_full):
    col = 0
    for llm in LLM_COLS:
        # arquivo 2
        if smell in f1_2 and llm in f1_2[smell]:
            f1_matrix[i, col] = f1_2[smell][llm]
            mcc_matrix[i, col] = mcc_2[smell][llm]
        col += 1

        # arquivo 1
        if smell in f1_1 and llm in f1_1[smell]:
            f1_matrix[i, col] = f1_1[smell][llm]
            mcc_matrix[i, col] = mcc_1[smell][llm]
        col += 1

# ==========================
# PLOT
# ==========================
plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 9,
})

#fig, axes = plt.subplots(1, 2, figsize=(8.6, 2.8), constrained_layout=True)
fig, axes = plt.subplots(2, 1, figsize=(5.2, 5.8), constrained_layout=True)

# --- F1 ---
im1 = axes[0].imshow(
    f1_matrix,
    vmin=0.0,
    vmax=1.0,
    aspect="auto",
    cmap="RdBu_r"
)
axes[0].set_xticks(range(len(x_labels)))
axes[0].set_xticklabels(x_labels)
axes[0].set_yticks(range(len(smells)))
axes[0].set_yticklabels(smells)
axes[0].set_xlabel("F1")

for i in range(f1_matrix.shape[0]):
    for j in range(f1_matrix.shape[1]):
        val = f1_matrix[i, j]
        if not np.isnan(val):
            axes[0].text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7)

cbar1 = fig.colorbar(im1, ax=axes[0], fraction=0.05, pad=0.02)
cbar1.ax.tick_params(labelsize=8)

# --- MCC ---
im2 = axes[1].imshow(
    mcc_matrix,
    vmin=-1.0,
    vmax=1.0,
    aspect="auto",
    cmap="RdBu_r"
)
axes[1].set_xticks(range(len(x_labels)))
axes[1].set_xticklabels(x_labels)
axes[1].set_yticks(range(len(smells)))
axes[1].set_yticklabels(smells)
axes[1].set_xlabel("MCC")

for i in range(mcc_matrix.shape[0]):
    for j in range(mcc_matrix.shape[1]):
        val = mcc_matrix[i, j]
        if not np.isnan(val):
            axes[1].text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7)

cbar2 = fig.colorbar(im2, ax=axes[1], fraction=0.05, pad=0.02)
cbar2.ax.tick_params(labelsize=8)

# linhas finas para separar células
for ax in axes:
    ax.set_xticks(np.arange(-0.5, len(x_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(smells), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)

# opcional: separadores visuais entre modelos
for ax in axes:
    for x in [1.5, 3.5]:
        ax.axvline(x=x, color="black", linewidth=0.8)

plt.savefig(OUTPUT_FILE_PNG, dpi=300, bbox_inches="tight")
plt.savefig(OUTPUT_FILE_PDF, bbox_inches="tight")
plt.close()

print("Saved:", OUTPUT_FILE_PNG)
print("Saved:", OUTPUT_FILE_PDF)