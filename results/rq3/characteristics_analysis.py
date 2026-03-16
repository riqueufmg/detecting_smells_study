from __future__ import annotations

from pathlib import Path
from itertools import combinations
import argparse
import pandas as pd

CATEGORIES = [
    "size",
    "coupling",
    "stability",
    "cohesion",
    "structural complexity",
]

DEFAULT_AUTHOR1 = "RQ3_author1 - samples.csv"
DEFAULT_AUTHOR2 = "RQ3_author2 - samples.csv"


def normalize_binary(value) -> int:
    if pd.isna(value):
        return 0
    if isinstance(value, (int, float)):
        return 0 if value == 0 else 1
    text = str(value).strip().lower()
    if text in {"", "0", "false", "no", "n", "nao", "não"}:
        return 0
    return 1


def load_single_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed:")], errors="ignore")

    required = {"id", "element", "smell", "llm", "approach"}.union(CATEGORIES)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path.name}: {missing}")

    for col in CATEGORIES:
        df[col] = df[col].apply(normalize_binary).astype(int)

    return df


def merge_authors(a1: pd.DataFrame, a2: pd.DataFrame, mode: str) -> pd.DataFrame:
    merged = a1.merge(a2, on="id", suffixes=("_a1", "_a2"), how="inner")
    if merged.empty:
        raise ValueError("No matching rows between author files using column 'id'.")

    result = pd.DataFrame()
    result["id"] = merged["id"]

    for meta in ["element", "smell", "llm", "approach", "detection", "justification"]:
        c1 = f"{meta}_a1"
        c2 = f"{meta}_a2"
        if c1 in merged.columns:
            result[meta] = merged[c1]
            if c2 in merged.columns:
                mismatch = merged[c1].astype(str) != merged[c2].astype(str)
                if mismatch.any() and meta in {"element", "smell", "llm", "approach"}:
                    print(f"[WARN] {mismatch.sum()} metadata mismatches in column '{meta}'. Using author1 values.")

    for col in CATEGORIES:
        c1 = merged[f"{col}_a1"]
        c2 = merged[f"{col}_a2"]
        if mode == "strict":
            result[col] = ((c1 == 1) & (c2 == 1)).astype(int)
        elif mode == "lenient":
            result[col] = ((c1 == 1) | (c2 == 1)).astype(int)
        elif mode == "author1":
            result[col] = c1.astype(int)
        elif mode == "author2":
            result[col] = c2.astype(int)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    return result


def category_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(df)
    for col in CATEGORIES:
        count = int(df[col].sum())
        rows.append({
            "category": col,
            "count": count,
            "percentage": round((count / n) * 100, 2) if n else 0,
        })
    return pd.DataFrame(rows).sort_values(["count", "category"], ascending=[False, True])


def combination_label(row: pd.Series) -> str:
    labels = [col for col in CATEGORIES if row[col] == 1]
    return " + ".join(labels) if labels else "none"


def combination_summary(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    combo = df.apply(combination_label, axis=1)
    out = combo.value_counts(dropna=False).reset_index()
    out.columns = ["combination", "count"]
    out["percentage"] = (out["count"] / len(df) * 100).round(2)
    return out.head(top_n)


def pair_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for a, b in combinations(CATEGORIES, 2):
        count = int(((df[a] == 1) & (df[b] == 1)).sum())
        rows.append({
            "pair": f"{a} + {b}",
            "count": count,
            "percentage": round((count / len(df)) * 100, 2) if len(df) else 0,
        })
    return pd.DataFrame(rows).sort_values(["count", "pair"], ascending=[False, True])


def breakdown_summary(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for group_value, group_df in df.groupby(group_col):
        n = len(group_df)
        for col in CATEGORIES:
            count = int(group_df[col].sum())
            rows.append({
                group_col: group_value,
                "category": col,
                "count": count,
                "percentage": round((count / n) * 100, 2) if n else 0,
                "n_items": n,
            })
    return pd.DataFrame(rows).sort_values([group_col, "count", "category"], ascending=[True, False, True])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize RQ3 labels for Section 5.3.2 (Architectural aspects used in LLM reasoning)."
    )
    parser.add_argument("--author1", default=DEFAULT_AUTHOR1, help="CSV file for author 1 annotations")
    parser.add_argument("--author2", default=DEFAULT_AUTHOR2, help="CSV file for author 2 annotations")
    parser.add_argument(
        "--mode",
        choices=["strict", "lenient", "author1", "author2"],
        default="author1",
        help=(
            "How to build the analysis dataset when two files are available: "
            "strict=both marked, lenient=either marked, author1=use author1 labels, author2=use author2 labels. "
            "For the paper, replace this with the final adjudicated labels when you finish disagreement resolution."
        ),
    )
    parser.add_argument("--outdir", default="rq3_5232_outputs", help="Output directory")
    args = parser.parse_args()

    author1_path = Path(args.author1)
    author2_path = Path(args.author2)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not author1_path.exists():
        raise FileNotFoundError(f"Author 1 file not found: {author1_path}")

    a1 = load_single_csv(author1_path)

    if author2_path.exists():
        a2 = load_single_csv(author2_path)
        analysis_df = merge_authors(a1, a2, args.mode)
        source_note = f"two authors merged with mode='{args.mode}'"
    else:
        analysis_df = a1.copy()
        source_note = "single author file (author2 not found)"
        print("[WARN] Author 2 file not found. Using only author 1 labels.")

    overall = category_summary(analysis_df)
    combos = combination_summary(analysis_df)
    pairs = pair_summary(analysis_df)
    by_smell = breakdown_summary(analysis_df, "smell")
    by_llm = breakdown_summary(analysis_df, "llm")
    by_approach = breakdown_summary(analysis_df, "approach")

    overall.to_csv(outdir / "overall_category_summary.csv", index=False, encoding="utf-8-sig")
    combos.to_csv(outdir / "top_combinations.csv", index=False, encoding="utf-8-sig")
    pairs.to_csv(outdir / "pair_cooccurrence.csv", index=False, encoding="utf-8-sig")
    by_smell.to_csv(outdir / "category_by_smell.csv", index=False, encoding="utf-8-sig")
    by_llm.to_csv(outdir / "category_by_llm.csv", index=False, encoding="utf-8-sig")
    by_approach.to_csv(outdir / "category_by_approach.csv", index=False, encoding="utf-8-sig")

    print("\n=== RQ3 / Section 5.3.2 Summary ===")
    print(f"Source: {source_note}")
    print(f"Instances: {len(analysis_df)}")
    print("\nOverall category usage:")
    print(overall.to_string(index=False))
    print("\nMost common label combinations:")
    print(combos.to_string(index=False))
    print("\nTop co-occurring pairs:")
    print(pairs.head(10).to_string(index=False))

    print("\nSaved files:")
    for name in [
        "overall_category_summary.csv",
        "top_combinations.csv",
        "pair_cooccurrence.csv",
        "category_by_smell.csv",
        "category_by_llm.csv",
        "category_by_approach.csv",
    ]:
        print(f"- {outdir / name}")


if __name__ == "__main__":
    main()
