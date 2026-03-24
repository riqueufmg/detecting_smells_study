from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
import pandas as pd


ARCH_COLS = [
    "size",
    "coupling",
    "stability",
    "cohesion",
    "structural complexity",
]

WEAK_REASONING_COL = "weak reasoning"
TEXT_COLS = ["id", "element", "detection", "smell", "llm", "approach", "justification"]


def normalize_binary(value) -> int:
    if pd.isna(value):
        return 0
    if isinstance(value, (int, float)):
        return 0 if value == 0 else 1

    text = str(value).strip().lower()
    positive = {"1", "x", "true", "yes", "y", "sim", "checked", "selected"}
    negative = {"0", "", "false", "no", "n", "nao", "não", "none", "null"}

    if text in positive:
        return 1
    if text in negative:
        return 0
    return 1


def normalize_approach(text: str) -> str:
    t = str(text).strip().lower()
    if "raw" in t:
        return "raw code"
    if any(s in t for s in ["metric", "dep", "dependency"]):
        return "metrics/deps"
    return str(text).strip()


def require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def load_dataset(path: Path, include_weak_reasoning: bool = False) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed:")], errors="ignore")

    required = TEXT_COLS + ARCH_COLS
    if include_weak_reasoning:
        required.append(WEAK_REASONING_COL)
    require_columns(df, required)

    df = df[required].copy()

    for col in ARCH_COLS:
        df[col] = df[col].apply(normalize_binary).astype(int)

    if include_weak_reasoning and WEAK_REASONING_COL in df.columns:
        df[WEAK_REASONING_COL] = df[WEAK_REASONING_COL].apply(normalize_binary).astype(int)

    df["approach"] = df["approach"].apply(normalize_approach)
    df["smell"] = df["smell"].astype(str).str.strip()
    df["llm"] = df["llm"].astype(str).str.strip()
    df["element"] = df["element"].astype(str).str.strip()
    df["justification"] = df["justification"].fillna("").astype(str).str.strip()

    return df


def merge_strict(author1: pd.DataFrame, author2: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    a1 = author1.copy()
    a2 = author2.copy()

    rename1 = {c: f"{c}_a1" for c in cols}
    rename2 = {c: f"{c}_a2" for c in cols}
    rename1.update({c: f"{c}_a1" for c in TEXT_COLS if c != "id"})
    rename2.update({c: f"{c}_a2" for c in TEXT_COLS if c != "id"})

    a1 = a1.rename(columns=rename1)
    a2 = a2.rename(columns=rename2)

    merged = pd.merge(a1, a2, on="id", how="inner")
    if len(merged) == 0:
        raise ValueError("No matching rows found between author1 and author2 using column 'id'.")

    out = pd.DataFrame()
    out["id"] = merged["id"]

    # Prefer author1 metadata, but keep author2 versions for checking mismatches
    for c in ["element", "detection", "smell", "llm", "approach", "justification"]:
        out[c] = merged[f"{c}_a1"]

    # Strict agreement on category columns
    for c in cols:
        out[c] = ((merged[f"{c}_a1"] == 1) & (merged[f"{c}_a2"] == 1)).astype(int)

    # Add mismatch report columns for optional inspection
    mismatch_rows = []
    for c in ["element", "detection", "smell", "llm", "approach"]:
        left = f"{c}_a1"
        right = f"{c}_a2"
        mismatches = (merged[left].astype(str) != merged[right].astype(str)).sum()
        mismatch_rows.append((c, int(mismatches)))
    mismatch_df = pd.DataFrame(mismatch_rows, columns=["field", "mismatch_count"])

    return out, mismatch_df


def active_categories(row: pd.Series, cols: list[str]) -> list[str]:
    return [c for c in cols if int(row[c]) == 1]


def combination_label(categories: list[str]) -> str:
    return "none" if not categories else " + ".join(categories)


def build_combination_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    out["active_categories"] = out.apply(lambda r: active_categories(r, cols), axis=1)
    out["n_categories"] = out["active_categories"].apply(len)
    out["combination"] = out["active_categories"].apply(combination_label)
    return out


def overall_summary(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    n = len(df)
    rows = []
    for col in cols:
        count = int(df[col].sum())
        rows.append({
            "category": col,
            "count": count,
            "percentage": round((count / n) * 100, 2) if n else 0.0,
            "n_instances": n,
        })
    return pd.DataFrame(rows).sort_values(["count", "category"], ascending=[False, True])


def summary_by_approach(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for approach, g in df.groupby("approach", dropna=False):
        n = len(g)
        for col in cols:
            count = int(g[col].sum())
            rows.append({
                "approach": approach,
                "category": col,
                "count": count,
                "percentage": round((count / n) * 100, 2) if n else 0.0,
                "n_instances": n,
            })
    return pd.DataFrame(rows).sort_values(["approach", "count", "category"], ascending=[True, False, True])


def combinations_by_approach(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    rows = []
    for approach, g in df.groupby("approach", dropna=False):
        n = len(g)
        counts = g["combination"].value_counts(dropna=False)
        for combo, count in counts.head(top_n).items():
            rows.append({
                "approach": approach,
                "combination": combo,
                "count": int(count),
                "percentage": round((count / n) * 100, 2) if n else 0.0,
                "n_instances": n,
            })
    return pd.DataFrame(rows).sort_values(["approach", "count", "combination"], ascending=[True, False, True])


def pair_cooccurrence_by_approach(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    rows = []
    for approach, g in df.groupby("approach", dropna=False):
        pair_counter = {}
        for _, row in g.iterrows():
            cats = row["active_categories"]
            for pair in combinations(cats, 2):
                pair_counter[pair] = pair_counter.get(pair, 0) + 1
        sorted_pairs = sorted(pair_counter.items(), key=lambda x: (-x[1], x[0]))
        for (a, b), count in sorted_pairs[:top_n]:
            rows.append({
                "approach": approach,
                "pair": f"{a} + {b}",
                "count": int(count),
                "percentage": round((count / len(g)) * 100, 2) if len(g) else 0.0,
                "n_instances": len(g),
            })
    if not rows:
        return pd.DataFrame(columns=["approach", "pair", "count", "percentage", "n_instances"])
    return pd.DataFrame(rows).sort_values(["approach", "count", "pair"], ascending=[True, False, True])


def summary_by_smell(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for smell, g in df.groupby("smell", dropna=False):
        n = len(g)
        for col in cols:
            count = int(g[col].sum())
            rows.append({
                "smell": smell,
                "category": col,
                "count": count,
                "percentage": round((count / n) * 100, 2) if n else 0.0,
                "n_instances": n,
            })
    return pd.DataFrame(rows).sort_values(["smell", "count", "category"], ascending=[True, False, True])


def summary_by_smell_and_approach(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for (smell, approach), g in df.groupby(["smell", "approach"], dropna=False):
        n = len(g)
        for col in cols:
            count = int(g[col].sum())
            rows.append({
                "smell": smell,
                "approach": approach,
                "category": col,
                "count": count,
                "percentage": round((count / n) * 100, 2) if n else 0.0,
                "n_instances": n,
            })
    return pd.DataFrame(rows).sort_values(
        ["smell", "approach", "count", "category"],
        ascending=[True, True, False, True]
    )


def smell_takeaways(df_smell_approach: pd.DataFrame, top_n: int = 2) -> pd.DataFrame:
    rows = []
    for (smell, approach), g in df_smell_approach.groupby(["smell", "approach"], dropna=False):
        top = g.sort_values(["count", "category"], ascending=[False, True]).head(top_n)
        cats = [f"{r['category']} ({int(r['count'])}; {r['percentage']}%)" for _, r in top.iterrows()]
        rows.append({
            "smell": smell,
            "approach": approach,
            "top_categories": " | ".join(cats)
        })
    return pd.DataFrame(rows).sort_values(["smell", "approach"])


def top_examples_by_smell(df: pd.DataFrame, top_k_per_smell: int = 6, max_just_len: int = 550) -> pd.DataFrame:
    rows = []
    for smell, g in df.groupby("smell", dropna=False):
        ranked = g.copy()
        ranked["just_len"] = ranked["justification"].str.len()
        ranked = ranked.sort_values(
            by=["n_categories", "just_len", "approach", "llm", "id"],
            ascending=[False, True, True, True, True]
        ).head(top_k_per_smell)
        for _, row in ranked.iterrows():
            just = row["justification"]
            if len(just) > max_just_len:
                just = just[: max_just_len - 3] + "..."
            rows.append({
                "smell": smell,
                "id": row["id"],
                "element": row["element"],
                "llm": row["llm"],
                "approach": row["approach"],
                "detection": row["detection"],
                "combination": row["combination"],
                "n_categories": row["n_categories"],
                "justification": just,
            })
    return pd.DataFrame(rows).sort_values(["smell", "n_categories", "approach", "llm"], ascending=[True, False, True, True])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate RQ3 qualitative analysis tables with strict dual-author mode.")
    parser.add_argument("--input", help="Single input CSV.")
    parser.add_argument("--author1", help="Author 1 CSV.")
    parser.add_argument("--author2", help="Author 2 CSV.")
    parser.add_argument("--mode", choices=["strict"], help="Dual-author merge mode.")
    parser.add_argument("--output-dir", default="rq3_reasoning_outputs", help="Directory for generated CSV files.")
    parser.add_argument("--include-weak-reasoning", action="store_true", help="Include weak reasoning as analysis category.")
    parser.add_argument("--top-combinations", type=int, default=10)
    parser.add_argument("--top-pairs", type=int, default=10)
    parser.add_argument("--top-examples-per-smell", type=int, default=6)
    return parser.parse_args()


def build_dataset_from_args(args: argparse.Namespace, cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame | None, str]:
    if args.input:
        df = load_dataset(Path(args.input), include_weak_reasoning=args.include_weak_reasoning)
        return df, None, f"single input: {args.input}"

    if args.author1 and args.author2 and args.mode == "strict":
        a1 = load_dataset(Path(args.author1), include_weak_reasoning=args.include_weak_reasoning)
        a2 = load_dataset(Path(args.author2), include_weak_reasoning=args.include_weak_reasoning)
        merged, mismatch_df = merge_strict(a1, a2, cols)
        return merged, mismatch_df, f"strict mode: {args.author1} + {args.author2}"

    raise ValueError("Use either --input FILE or --author1 FILE --author2 FILE --mode strict.")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cols = ARCH_COLS.copy()
    if args.include_weak_reasoning:
        cols.append(WEAK_REASONING_COL)

    df, mismatch_df, source_desc = build_dataset_from_args(args, cols)
    df = build_combination_columns(df, cols)

    overall = overall_summary(df, cols)
    by_approach = summary_by_approach(df, cols)
    combos_by_approach = combinations_by_approach(df, top_n=args.top_combinations)
    pairs_by_approach = pair_cooccurrence_by_approach(df, top_n=args.top_pairs)
    by_smell = summary_by_smell(df, cols)
    by_smell_approach = summary_by_smell_and_approach(df, cols)
    takeaways = smell_takeaways(by_smell_approach)
    examples = top_examples_by_smell(df, top_k_per_smell=args.top_examples_per_smell)

    df.to_csv(output_dir / "dataset_used.csv", index=False, encoding="utf-8-sig")
    overall.to_csv(output_dir / "overall_category_summary.csv", index=False, encoding="utf-8-sig")
    by_approach.to_csv(output_dir / "categories_by_approach.csv", index=False, encoding="utf-8-sig")
    combos_by_approach.to_csv(output_dir / "combinations_by_approach.csv", index=False, encoding="utf-8-sig")
    pairs_by_approach.to_csv(output_dir / "pairs_by_approach.csv", index=False, encoding="utf-8-sig")
    by_smell.to_csv(output_dir / "categories_by_smell.csv", index=False, encoding="utf-8-sig")
    by_smell_approach.to_csv(output_dir / "categories_by_smell_and_approach.csv", index=False, encoding="utf-8-sig")
    takeaways.to_csv(output_dir / "smell_takeaways.csv", index=False, encoding="utf-8-sig")
    examples.to_csv(output_dir / "representative_examples_by_smell.csv", index=False, encoding="utf-8-sig")
    if mismatch_df is not None:
        mismatch_df.to_csv(output_dir / "metadata_mismatches_between_authors.csv", index=False, encoding="utf-8-sig")

    print(f"Source: {source_desc}")
    print(f"Total instances: {len(df)}")
    print(f"Output directory: {output_dir.resolve()}")
    print("\n=== Overall category summary ===")
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
