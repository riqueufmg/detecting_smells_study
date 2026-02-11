import csv
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt

LLM = "qwen"
BASE_DIR = Path("data/processed")
INPUT_CSV = BASE_DIR / "results" / LLM / "all_samples_details.csv"
PLOTS_DIR = BASE_DIR / "results" / LLM / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

ORDER = ["tp", "tn", "fp", "fn"]
SMELLS = ["god_component", "unstable_dependency", "insufficient_modularization", "hublike_modularization"]


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
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    # Load all rows once
    rows = list(read_rows(INPUT_CSV))

    for smell in SMELLS:
        groups = defaultdict(list)  # classification -> list[context_size]

        for r in rows:
            if r.get("smell") != smell:
                continue
            cls = (r.get("classification") or "").strip().lower()
            ctx = to_int(r.get("context_size"), default=None)
            if cls in ORDER and ctx is not None and ctx >= 0:
                groups[cls].append(ctx)

        data = [groups[c] for c in ORDER]

        plt.figure(figsize=(9, 5))
        plt.boxplot(data, labels=ORDER, showfliers=True)  # outliers on

        for i, c in enumerate(ORDER, start=1):
            ys = groups[c]
            xs = [i] * len(ys)
            plt.scatter(xs, ys, s=20, alpha=0.6)

        plt.xlabel("classification")
        plt.ylabel("context size")
        plt.title(f"{LLM} — {smell}")

        out_path = PLOTS_DIR / f"{LLM}_{smell}_boxplot_context_by_classification.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()

        print("Saved:", out_path)


if __name__ == "__main__":
    main()
