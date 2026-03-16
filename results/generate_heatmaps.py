import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import math

# ==========================
# CONFIG
# ==========================
INPUT_FILE = "metrics_results.csv"
#INPUT_FILE = "code_results.csv"
OUTPUT_FILE_PNG = "heatmap_f1_mcc_clean.png"
OUTPUT_FILE_PDF = "heatmap_f1_mcc_clean.pdf"

SMELL_COL = "smell"
GROUND_TRUTH_COL = "designite"
LLM_COLS = ["gpt", "deepseek", "qwen"]

# Smell abbreviations (ajuste se seus nomes vierem com variações)
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
    return SMELL_ABBR.get(name, name)  # fallback: mantém o original se não achar

# ==========================
# LOAD + COMPUTE
# ==========================
delim = sniff_delimiter(INPUT_FILE)
df = pd.read_csv(INPUT_FILE, sep=delim, encoding="utf-8-sig")
df.columns = [c.strip().lower() for c in df.columns]

df[GROUND_TRUTH_COL] = to_binary_series(df[GROUND_TRUTH_COL])
for llm in LLM_COLS:
    df[llm] = to_binary_series(df[llm])

smells_full = sorted(df[SMELL_COL].astype(str).unique())
smells = [abbreviate_smell(s) for s in smells_full]

f1_matrix = np.zeros((len(smells_full), len(LLM_COLS)))
mcc_matrix = np.zeros((len(smells_full), len(LLM_COLS)))

for i, smell in enumerate(smells_full):
    sub = df[df[SMELL_COL].astype(str) == smell]
    gt = sub[GROUND_TRUTH_COL]
    for j, llm in enumerate(LLM_COLS):
        tp, tn, fp, fn = compute_confusion(gt, sub[llm])
        f1_matrix[i, j] = compute_f1(tp, fp, fn)
        mcc_matrix[i, j] = compute_mcc(tp, tn, fp, fn)

# ==========================
# PLOT (paper-friendly)
# ==========================
plt.rcParams.update({
    "font.size": 9,          # base
    "axes.labelsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), constrained_layout=True)

# --- F1 ---
im1 = axes[0].imshow(f1_matrix, vmin=0.0, vmax=1.0, aspect="auto", cmap="RdBu_r")
axes[0].set_xticks(range(len(LLM_COLS)))
axes[0].set_xticklabels(LLM_COLS, rotation=30, ha="right")
axes[0].set_yticks(range(len(smells)))
axes[0].set_yticklabels(smells)
axes[0].set_xlabel("F1")
#axes[0].set_ylabel("Smell")

# annotate (smaller font)
for i in range(f1_matrix.shape[0]):
    for j in range(f1_matrix.shape[1]):
        axes[0].text(j, i, f"{f1_matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)

cbar1 = fig.colorbar(im1, ax=axes[0], fraction=0.05, pad=0.02)
cbar1.ax.tick_params(labelsize=8)
#cbar1.set_label("F1", fontsize=9)

# --- MCC ---
im2 = axes[1].imshow(mcc_matrix, vmin=-1.0, vmax=1.0, aspect="auto", cmap="RdBu_r")
axes[1].set_xticks(range(len(LLM_COLS)))
axes[1].set_xticklabels(LLM_COLS, rotation=30, ha="right")
axes[1].set_yticks(range(len(smells)))
axes[1].set_yticklabels(smells)
axes[1].set_xlabel("MCC")
axes[1].set_ylabel("")  # evita repetir "Smell" do lado direito

for i in range(mcc_matrix.shape[0]):
    for j in range(mcc_matrix.shape[1]):
        axes[1].text(j, i, f"{mcc_matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)

cbar2 = fig.colorbar(im2, ax=axes[1], fraction=0.05, pad=0.02)
cbar2.ax.tick_params(labelsize=8)
#cbar2.set_label("MCC", fontsize=9)

# Sem título (você usa caption no LaTeX)
plt.savefig(OUTPUT_FILE_PNG, dpi=300, bbox_inches="tight")
#plt.savefig(OUTPUT_FILE_PDF, bbox_inches="tight")
plt.close()

print("Saved:", OUTPUT_FILE_PNG)
print("Saved:", OUTPUT_FILE_PDF)