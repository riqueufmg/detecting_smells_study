from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score

# =========================
# CONFIGURAÇÃO
# =========================
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


# =========================
# FUNÇÕES AUXILIARES
# =========================
def normalize_binary(value):
    """
    Converte diferentes formas de marcação para 0 ou 1.
    Ajuste aqui se suas planilhas tiverem algum padrão específico.
    """
    if pd.isna(value):
        return 0

    # Se já for número
    if isinstance(value, (int, float, np.integer, np.floating)):
        return 1 if value != 0 else 0

    # Se for texto
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

    # Qualquer texto não vazio conta como presença da categoria
    return 1


def load_and_prepare(file_path, suffix):
    df = pd.read_csv(file_path)

    # Remove colunas totalmente vazias ou lixo do Excel
    df = df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed:")], errors="ignore")

    # Verificações mínimas
    required = [ID_COLUMN] + CATEGORY_COLUMNS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Arquivo {file_path} está sem colunas obrigatórias: {missing}")

    # Mantém colunas úteis
    metadata_cols = [c for c in ["element", "detection", "smell", "llm", "approach", "justification"] if c in df.columns]
    keep_cols = [ID_COLUMN] + metadata_cols + CATEGORY_COLUMNS
    df = df[keep_cols].copy()

    # Normaliza categorias para 0/1
    for col in CATEGORY_COLUMNS:
        df[col] = df[col].apply(normalize_binary).astype(int)

    # Renomeia colunas categóricas e metadados
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
    """
    Retorna:
    a = ambos 1
    b = autor1 1 / autor2 0
    c = autor1 0 / autor2 1
    d = ambos 0
    """
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


# =========================
# CARGA DOS DADOS
# =========================
author1 = load_and_prepare(AUTHOR1_FILE, "A1")
author2 = load_and_prepare(AUTHOR2_FILE, "A2")

# Merge pelo ID
merged = pd.merge(author1, author2, on=ID_COLUMN, how="inner")

if len(merged) == 0:
    raise ValueError("Nenhuma linha foi casada entre os dois arquivos pelo campo 'id'.")

print(f"Total de instâncias pareadas: {len(merged)}")

# Checagem opcional de consistência de metadados
for meta in ["element", "smell", "llm", "approach"]:
    c1 = f"{meta}_A1"
    c2 = f"{meta}_A2"
    if c1 in merged.columns and c2 in merged.columns:
        mismatches = merged[merged[c1].astype(str) != merged[c2].astype(str)]
        if len(mismatches) > 0:
            print(f"[AVISO] Há {len(mismatches)} divergências na coluna de metadado '{meta}' entre os dois arquivos.")


# =========================
# KAPPA POR CATEGORIA
# =========================
results = []
contingency_rows = []

for category in CATEGORY_COLUMNS:
    col_a1 = f"{category}_A1"
    col_a2 = f"{category}_A2"

    y1 = merged[col_a1]
    y2 = merged[col_a2]

    # Cohen's Kappa
    kappa = cohen_kappa_score(y1, y2)

    # Concordância observada
    po = observed_agreement(y1, y2)

    # Frequências
    prevalence_a1 = float(y1.mean())
    prevalence_a2 = float(y2.mean())

    # Tabela de contingência
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

# Kappa médio simples entre categorias
mean_kappa = results_df["kappa"].mean()

print("\n=== COHEN'S KAPPA POR CATEGORIA ===")
print(results_df.to_string(index=False))

print(f"\nKappa médio entre categorias: {mean_kappa:.4f}")

print("\n=== TABELAS DE CONTINGÊNCIA ===")
print(contingency_df.to_string(index=False))

# Salva em CSV
results_df.to_csv(OUTPUT_RESULTS, index=False, encoding="utf-8-sig")
contingency_df.to_csv(OUTPUT_CONTINGENCY, index=False, encoding="utf-8-sig")

print(f"\nArquivos gerados:")
print(f"- {OUTPUT_RESULTS}")
print(f"- {OUTPUT_CONTINGENCY}")