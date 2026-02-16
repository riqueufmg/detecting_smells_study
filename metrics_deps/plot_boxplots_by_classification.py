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

    # load llm output file (prediction)
    content = llm_file.read_text(encoding="utf-8", errors="replace").strip()
    if not content:
        return False

    # convert the content to JSON format
    try:
        data = json.loads(content)
        return bool(data.get("detection", False))
    except json.JSONDecodeError:
        pass

    # search for first JSON block, to avoid trash data
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return False

    # return detection result (true/false)
    try:
        data = json.loads(content[start : end + 1])
        return bool(data.get("detection", False))
    except json.JSONDecodeError:
        return False

# utility function to classify sample
def classify(pred: int, gold: int) -> str:
    if pred == 1 and gold == 1:
        return "tp"
    if pred == 0 and gold == 0:
        return "tn"
    if pred == 1 and gold == 0:
        return "fp"
    return "fn"

def candidate_llm_filenames(prompt_path_str: str, pkg: str, cls: str) -> list[str]:
    """
    Generate possible output filenames.
    Your dataset transitioned from underscore naming to dot naming, and inner classes also changed.
    We'll try a small set of candidates and pick the first one that exists.
    """
    p = Path(prompt_path_str)
    name = p.name  # original basename from CSV prompt_file

    candidates = [name]

    # 1) underscore <-> dot variants
    # org_apache_commons_lang3_math.txt -> org.apache.commons.lang3.math.txt
    candidates.append(Path(name).stem.replace("_", ".") + ".txt")

    # 2) package/class driven variants (handles inner-class naming)
    # - package-level: org.apache.commons.io.channels.txt
    # - class-level:   org.jsoup.nodes_Attributes.Dataset.txt (outer pkg + "_" + outer + "." + inner)
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

    # de-duplicate while preserving order
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

    # output folder for combined plots (all llms in the same figure)
    OUT_DIR = BASE_DIR / "results" / "plots" / "all_llms"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # iteract by smell/sample files
    for smell, csv_name in SMELL_FILES.items():

        # build sample path
        csv_path = CANDIDATES_DIR / csv_name
        if not csv_path.exists():
            print(f"[SKIP] missing csv: {csv_path}")
            continue

        # store results per LLM and classification
        # example key: ("gpt","tp") -> [ctx1, ctx2, ...]
        groups = defaultdict(list)

        # counters per LLM
        missing_llm = {llm: 0 for llm in LLMS}
        total = 0

        # read samples csv file
        with csv_path.open("r", encoding="utf-8", newline="") as f:

            reader = csv.DictReader(f)

            # iteract instances
            for row in reader:
                total += 1 # counter

                label = int(row["label"]) # load Designite classification
                ctx = int(row["context_size"]) # load prompt context size

                repo = row["project"].strip() # load repo name
                prompt_file = row["prompt_file"] # load prompt file path (string)

                pkg = (row.get("package") or "").strip()
                cls_name = (row.get("class") or "").strip()

                # for each LLM, load its prediction and store ctx in proper (LLM, tp/tn/fp/fn)
                for LLM in LLMS:

                    # build path to LLM's output (repo-based layout)
                    llm_dir = LLM_OUTPUTS_DIR / repo / smell / LLM

                    # try multiple possible basenames
                    llm_file = None
                    for candidate_name in candidate_llm_filenames(prompt_file, pkg, cls_name):
                        p = llm_dir / candidate_name
                        if p.exists():
                            llm_file = p
                            break

                    if llm_file is None:
                        missing_llm[LLM] += 1
                        continue

                    # load LLM prediction (true/false)
                    if load_detection(llm_file):
                        pred = 1
                    else:
                        pred = 0

                    # compare LLM prediction w/ Designite result and return classification (tp, tn, fp, fn)
                    cls = classify(pred, label)

                    # add context size into a class list
                    groups[(LLM, cls)].append(ctx)

        # --- build plot data: 4 blocks (tp/tn/fp/fn), each block with 3 boxes (gpt/deepseek/qwen) ---
        data = []
        tick_labels = []
        positions = []

        pos = 1
        block_gap = 1.5  # space between classification blocks

        for c in ORDER:
            for llm in LLMS:
                data.append(groups[(llm, c)])
                tick_labels.append(f"{c}\n{llm}")  # two-line label
                positions.append(pos)
                pos += 1
            pos += block_gap  # gap between tp/tn/fp/fn blocks

        # plot
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

        # show points to avoid “single instance line” confusion
        # (keeps your idea; just shifted to the custom positions)
        for i, ys in enumerate(data):
            x = positions[i]
            xs = [x] * len(ys)
            plt.scatter(xs, ys, s=14, alpha=0.45)

        # save image
        out_path = OUT_DIR / f"{smell}_boxplot_context_by_classification_all_llms_grouped_by_class.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()

        # print summary
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
