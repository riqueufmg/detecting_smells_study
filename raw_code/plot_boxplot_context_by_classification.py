import csv
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt

def read_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row

def to_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default

def main():

    LLMS = ["gpt", "deepseek", "qwen"]

    BASE_DIR = Path("data/processed")

    # input CSV per LLM (generated previously)
    INPUT_CSVS = {
        llm: (BASE_DIR / "results" / llm / "all_samples_details.csv")
        for llm in LLMS
    }

    # output folder for combined plots (all llms in the same figure)
    PLOTS_DIR = BASE_DIR / "results" / "plots" / "all_llms_from_details"
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    ORDER = ["tp", "tn", "fp", "fn"]
    SMELLS = ["god_component", "unstable_dependency", "insufficient_modularization", "hublike_modularization"]

    # Load all rows once (for all LLMs)
    all_rows = []
    for llm, path in INPUT_CSVS.items():
        if not path.exists():
            raise FileNotFoundError(f"Input CSV not found: {path}")
        for r in read_rows(path):
            r["_llm"] = llm  # keep which LLM generated this row
            all_rows.append(r)

    # Build one plot per smell, showing ALL LLMs grouped by classification (tp/tn/fp/fn)
    for smell in SMELLS:

        # groups key: (classification, llm) -> list[context_size]
        groups = defaultdict(list)

        # count per llm (useful sanity check)
        counts = defaultdict(int)

        for r in all_rows:
            if r.get("smell") != smell:
                continue

            llm = r.get("_llm")
            cls = (r.get("classification") or "").strip().lower()
            ctx = to_int(r.get("context_size"), default=None)

            if llm in LLMS and cls in ORDER and ctx is not None and ctx >= 0:
                groups[(cls, llm)].append(ctx)
                counts[(cls, llm)] += 1

        # --- build data in the preferred order:
        # tp(gpt,deepseek,qwen), tn(...), fp(...), fn(...)
        data = []
        tick_labels = []
        positions = []

        pos = 1
        block_gap = 1.5  # space between tp/tn/fp/fn blocks

        for cls in ORDER:
            for llm in LLMS:
                data.append(groups[(cls, llm)])
                tick_labels.append(f"{cls}\n{llm}")  # two-line label
                positions.append(pos)
                pos += 1
            pos += block_gap

        plt.figure(figsize=(12, 5))

        # NOTE: labels -> tick_labels to avoid MatplotlibDeprecationWarning
        plt.boxplot(
            data,
            positions=positions,
            tick_labels=tick_labels,
            showfliers=True
        )

        # optional scatter points (helpful when some groups have few samples)
        for i, ys in enumerate(data):
            x = positions[i]
            xs = [x] * len(ys)
            plt.scatter(xs, ys, s=20, alpha=0.6)

        plt.xlabel("classification / LLM")
        plt.ylabel("context size")
        plt.title(f"{smell}: context_size by TP/TN/FP/FN (all LLMs)")

        out_path = PLOTS_DIR / f"{smell}_boxplot_context_by_classification_all_llms.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()

        # print a compact summary of counts
        parts = []
        for cls in ORDER:
            parts.append(
                f"{cls}={{" + ", ".join(f"{llm}:{len(groups[(cls,llm)])}" for llm in LLMS) + "}}"
            )
        print(f"Saved: {out_path} | " + " ".join(parts))

if __name__ == "__main__":
    main()
