import csv
import re
from pathlib import Path


BASE_DIR = Path("data/processed")
CANDIDATES_DIR = BASE_DIR / "results" / "candidates_sampled"

# Prompts reais (pela sua screenshot)
PROMPTS_ROOT = BASE_DIR / "prompts" / "smell_detection"

SMELL_FILES = {
    "god_component": "god_component_sample.csv",
    "unstable_dependency": "unstable_dependency_sample.csv",
    "insufficient_modularization": "insufficient_modularization_sample.csv",
    "hublike_modularization": "hublike_modularization_sample.csv",
}

HEADER_RE = re.compile(r"^##CONTEXT_SIZE=(\d+)\s*$")


def read_context_size(prompt_path: Path) -> int:
    with prompt_path.open("r", encoding="utf-8", errors="replace") as f:
        first = f.readline().strip()
    m = HEADER_RE.match(first)
    if not m:
        raise ValueError(f"Missing/invalid CONTEXT_SIZE header: {prompt_path}")
    return int(m.group(1))


def candidate_paths(smell_dir: str, repo: str, row: dict) -> list[Path]:
    """
    Build a list of candidate prompt paths for a given CSV row.

    Your real naming patterns:
    - GC/UD (package-level):   com.service.usage2.txt
    - IM/HM (class-level):    com.service.definition_ServiceDefinition.txt
    - IM/HM (inner class):    org.apache.commons.lang3.time_FastDatePrinter.StringLiteral.txt
                               where CSV has:
                                 package = org.apache.commons.lang3.time.FastDatePrinter
                                 class   = StringLiteral

    CSV 'prompt_file' may still reference an old naming with underscores.
    """
    folder = PROMPTS_ROOT / smell_dir / repo
    out: list[Path] = []

    prompt_file = (row.get("prompt_file") or "").strip()
    pkg = (row.get("package") or "").strip()
    cls = (row.get("class") or "").strip()

    # 1) Try using prompt_file path as-is and as basename under the correct folder
    if prompt_file:
        p = Path(prompt_file)
        out.append(p)  # absolute/relative as recorded
        out.append(folder / p.name)  # same basename in correct folder

        # Convert old underscore naming basename -> dot naming basename
        # e.g., org_apache_commons_lang3_time_FastDatePrinter_StringLiteral.txt
        # -> org.apache.commons.lang3.time.FastDatePrinter.StringLiteral.txt (not always exact)
        stem = p.stem
        out.append(folder / f"{stem.replace('_', '.')}.txt")

    # 2) Build from (package, class) according to smell level naming
    if pkg:
        pkg_dot = pkg
        pkg_under = pkg.replace(".", "_")

        if cls:
            # ---- Inner class special-case ----
            # CSV: package = outer class FQN (outer_pkg.OuterClass)
            #      class  = InnerClass
            # Disk: outer_pkg_OuterClass.InnerClass.txt
            if "." in pkg_dot:
                parts = pkg_dot.split(".")
                outer_class = parts[-1]
                outer_pkg = ".".join(parts[:-1])
                if outer_pkg:
                    out.append(folder / f"{outer_pkg}_{outer_class}.{cls}.txt")

            # ---- Other plausible patterns ----
            # class-level: package_or_outerfqn + "_" + class
            out.append(folder / f"{pkg_dot}_{cls}.txt")
            out.append(folder / f"{pkg_under}_{cls}.txt")

            # class-level with dot separator
            out.append(folder / f"{pkg_dot}.{cls}.txt")
            out.append(folder / f"{pkg_under}.{cls}.txt")

        else:
            # package-level: package.txt
            out.append(folder / f"{pkg_dot}.txt")
            out.append(folder / f"{pkg_under}.txt")

    # unique preserving order
    seen = set()
    uniq = []
    for p in out:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def resolve_prompt_path(smell_dir: str, repo: str, row: dict) -> Path | None:
    folder = PROMPTS_ROOT / smell_dir / repo
    if not folder.exists():
        return None

    # First: try direct candidates
    for p in candidate_paths(smell_dir, repo, row):
        if p.exists():
            return p

    # Controlled search fallback (avoid picking wrong file)
    pkg = (row.get("package") or "").strip()
    cls = (row.get("class") or "").strip()

    # Class-level: try exact inner-class pattern first again (fast)
    if pkg and cls:
        # inner class exact
        if "." in pkg:
            parts = pkg.split(".")
            outer_class = parts[-1]
            outer_pkg = ".".join(parts[:-1])
            if outer_pkg:
                inner_name = folder / f"{outer_pkg}_{outer_class}.{cls}.txt"
                if inner_name.exists():
                    return inner_name

        # If still not found: try unique "*_Class.txt"
        matches = list(folder.glob(f"*_{cls}.txt"))
        if len(matches) == 1:
            return matches[0]
        return None  # none or ambiguous

    # Package-level: exact package.txt, else accept unique prefix match "{pkg}_*.txt"
    if pkg and not cls:
        exact = folder / f"{pkg}.txt"
        if exact.exists():
            return exact

        prefix_matches = list(folder.glob(f"{pkg}_*.txt"))
        if len(prefix_matches) == 1:
            return prefix_matches[0]

    return None


def update_one_csv(smell_dir: str, csv_path: Path) -> dict:
    tmp = csv_path.with_suffix(csv_path.suffix + ".tmp")

    total = updated = unchanged = missing_prompt = bad_header = 0

    with csv_path.open("r", encoding="utf-8", newline="") as fin:
        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames or []

        if "context_size" not in fieldnames:
            raise ValueError(f"'context_size' column not found in {csv_path}")
        if "project" not in fieldnames:
            raise ValueError(f"'project' column not found in {csv_path}")

        with tmp.open("w", encoding="utf-8", newline="") as fout:
            writer = csv.DictWriter(fout, fieldnames=fieldnames)
            writer.writeheader()

            for row in reader:
                total += 1
                repo = (row.get("project") or "").strip()

                prompt_path = resolve_prompt_path(smell_dir, repo, row)
                if prompt_path is None:
                    missing_prompt += 1
                    print(
                        "[MISSING]",
                        f"smell={smell_dir}",
                        f"repo={repo}",
                        f"package={row.get('package','')}",
                        f"class={row.get('class','')}",
                        f"prompt_file={row.get('prompt_file','')}",
                    )
                    writer.writerow(row)  # unchanged
                    continue

                try:
                    new_ctx = read_context_size(prompt_path)
                except ValueError:
                    bad_header += 1
                    print("[BAD_HEADER]", prompt_path)
                    writer.writerow(row)
                    continue

                old_ctx = (row.get("context_size") or "").strip()
                if old_ctx == str(new_ctx):
                    unchanged += 1
                else:
                    row["context_size"] = str(new_ctx)
                    updated += 1

                writer.writerow(row)

    tmp.replace(csv_path)

    return {
        "file": csv_path.name,
        "smell": smell_dir,
        "total": total,
        "updated": updated,
        "unchanged": unchanged,
        "missing_prompt": missing_prompt,
        "bad_header": bad_header,
    }


def main():
    if not PROMPTS_ROOT.exists():
        raise FileNotFoundError(f"Prompts root not found: {PROMPTS_ROOT}")
    if not CANDIDATES_DIR.exists():
        raise FileNotFoundError(f"Candidates dir not found: {CANDIDATES_DIR}")

    for smell_dir, csv_name in SMELL_FILES.items():
        csv_path = CANDIDATES_DIR / csv_name
        stats = update_one_csv(smell_dir, csv_path)
        print(
            f"[OK] {stats['file']} ({stats['smell']}): total={stats['total']} "
            f"updated={stats['updated']} unchanged={stats['unchanged']} "
            f"missing_prompt={stats['missing_prompt']} bad_header={stats['bad_header']}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
