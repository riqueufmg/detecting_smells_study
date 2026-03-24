import csv
import json
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt


SMELL_FILES = {
    "god_component": "god_component_sample.csv",
    "unstable_dependency": "unstable_dependency_sample.csv",
    "insufficient_modularization": "insufficient_modularization_sample.csv",
    "hublike_modularization": "hublike_modularization_sample.csv",
}

ORDER = ["tp", "tn", "fp", "fn"]

def load_detection(llm_file: Path) -> bool:
    if not llm_file.exists():
        return False

    content = llm_file.read_text(encoding="utf-8", errors="replace").strip()
    if not content:
        return False

    try:
        data = json.loads(content)
        return bool(data.get("detection", False))
    except json.JSONDecodeError:
        pass

    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return False

    try:
        data = json.loads(content[start : end + 1])
        return bool(data.get("detection", False))
    except json.JSONDecodeError:
        return False

def classify(pred: int, gold: int) -> str:
    if pred == 1 and gold == 1:
        return "tp"
    if pred == 0 and gold == 0:
        return "tn"
    if pred == 1 and gold == 0:
        return "fp"
    return "fn"

def candidate_llm_filenames(prompt_path_str: str, pkg: str, cls: str) -> list[str]:

    p = Path(prompt_path_str)
    name = p.name  # original basename from CSV prompt_file

    candidates = [name]

    candidates.append(Path(name).stem.replace("_", ".") + ".txt")

    if pkg:
        if cls:
            # Try simple pkg_cls (old style)
            candidates.append(f"{pkg}_{cls}.txt")

            # Inner-class style: outer_pkg_OuterClass.InnerClass.txt
            # If pkg looks like "org.jsoup.nodes.Attributes" and cls = "Dataset"
            # -> org.jsoup.nodes_Attributes.Dataset.txt
            if "." in pkg:
                parts = pkg.split(".")
                outer = parts[-1]
                outer_pkg = ".".join(parts[:-1])
                if outer_pkg:
                    candidates.append(f"{outer_pkg}_{outer}.{cls}.txt")
        else:
            candidates.append(f"{pkg}.txt")

    seen = set()
    uniq = []
    for c in candidates:
        if c not in seen:
            uniq.append(c)
            seen.add(c)
    return uniq

def main():

    LLMS = ["gpt", "deepseek", "qwen"]
    BASE_DIR = Path("data/processed")
    CANDIDATES_DIR = BASE_DIR / "results" / "candidates_sampled"
    LLM_OUTPUTS_DIR = BASE_DIR / "llm_outputs"

    OUT_DIR = BASE_DIR / "results" / "plots" / "all_llms"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for smell, csv_name in SMELL_FILES.items():

        csv_path = CANDIDATES_DIR / csv_name
        if not csv_path.exists():
            print(f"[SKIP] missing csv: {csv_path}")
            continue

        groups = defaultdict(list)

        missing_llm = {llm: 0 for llm in LLMS}
        total = 0

        with csv_path.open("r", encoding="utf-8", newline="") as f:

            reader = csv.DictReader(f)

            for row in reader:
                total += 1

                label = int(row["label"]) 
                ctx = int(row["context_size"])

                repo = row["project"].strip()
                prompt_file = row["prompt_file"]

                pkg = (row.get("package") or "").strip()
                cls_name = (row.get("class") or "").strip()

                for LLM in LLMS:

                    llm_dir = LLM_OUTPUTS_DIR / repo / smell / LLM

                    llm_file = None
                    for candidate_name in candidate_llm_filenames(prompt_file, pkg, cls_name):
                        p = llm_dir / candidate_name
                        if p.exists():
                            llm_file = p
                            break

                    if llm_file is None:
                        missing_llm[LLM] += 1
                        continue

                    if load_detection(llm_file):
                        pred = 1
                    else:
                        pred = 0

                    cls = classify(pred, label)

                    # add context size into a class list
                    groups[(LLM, cls)].append(ctx)

        data = []
        tick_labels = []
        positions = []

        pos = 1
        block_gap = 1.5

        for c in ORDER:
            for llm in LLMS:
                data.append(groups[(llm, c)])
                tick_labels.append(f"{c}\n{llm}")  # two-line label
                positions.append(pos)
                pos += 1
            pos += block_gap

        plt.figure(figsize=(12, 5))
        plt.boxplot(
            data,
            positions=positions,
            tick_labels=tick_labels,
            showfliers=True
        )
        plt.xlabel("classification / LLM")
        plt.ylabel("context size")
        plt.title(f"{smell}: context_size by TP/TN/FP/FN (all LLMs)")

        for i, ys in enumerate(data):
            x = positions[i]
            xs = [x] * len(ys)
            plt.scatter(xs, ys, s=14, alpha=0.45)

        out_path = OUT_DIR / f"{smell}_boxplot_context.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()

        counts_str = []
        for llm in LLMS:
            counts_str.append(
                f"{llm}={{" + ", ".join(f"{cc}:{len(groups[(llm,cc)])}" for cc in ORDER) + "}}"
            )

        print(
            f"[OK] {smell}: saved {out_path} | total={total} "
            f"missing_llm={{{', '.join(f'{k}:{v}' for k,v in missing_llm.items())}}} "
            f"counts=" + " ".join(counts_str)
        )

if __name__ == "__main__":
    main()
