import csv
import json
from pathlib import Path
from typing import Dict, Set, List, Any

BASE_IN = Path("data/processed/consolidated_detection")
BASE_OUT = Path("data/processed/results/consolidated_detection")
CANDIDATES_CSV = Path("data/processed/results/candidates_sampled/god_component_sample.csv")

ENGINES = ["gpt", "deepseek", "qwen"]
FILES = ["god_component_llm.json", "god_component_designite.json"]

def read_candidates(csv_path: Path) -> Dict[str, Set[str]]:
    """
    Returns: { project_name -> set(packages in candidates_sampled) }
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Candidates CSV not found: {csv_path}")

    proj_pkgs: Dict[str, Set[str]] = {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("smell") != "God Component":
                continue
            project = (row.get("project") or "").strip()
            pkg = (row.get("package") or "").strip()
            if not project or not pkg:
                continue
            proj_pkgs.setdefault(project, set()).add(pkg)
    return proj_pkgs

def normalize_packages_field(pkg_field: Any) -> List[str]:
    """
    Handles cases where entry["package"] might be a string or a list.
    """
    if isinstance(pkg_field, list):
        return [str(p).strip() for p in pkg_field if str(p).strip()]
    if pkg_field is None:
        return []
    return [str(pkg_field).strip()] if str(pkg_field).strip() else []

def filter_json_entries(entries: List[dict], allowed: Set[str]) -> List[dict]:
    kept = []
    for e in entries:
        pkgs = normalize_packages_field(e.get("package"))
        # keep entry if ANY package in it is in allowed set
        if any(p in allowed for p in pkgs):
            kept.append(e)
    return kept

def main():
    proj_pkgs = read_candidates(CANDIDATES_CSV)
    if not BASE_IN.exists():
        raise FileNotFoundError(f"Input base not found: {BASE_IN}")

    BASE_OUT.mkdir(parents=True, exist_ok=True)

    projects = sorted([p.name for p in BASE_IN.iterdir() if p.is_dir()])
    print(f"Found {len(projects)} projects in {BASE_IN}: {projects}")

    total_written = 0

    for project in projects:
        allowed = proj_pkgs.get(project, set())
        if not allowed:
            print(f"[SKIP] {project}: no packages found in candidates CSV")
            continue

        for engine in ENGINES:
            in_dir = BASE_IN / project / "god_component" / engine
            if not in_dir.exists():
                print(f"[SKIP] {project}/{engine}: input dir not found: {in_dir}")
                continue

            out_dir = BASE_OUT / project / "god_component" / engine
            out_dir.mkdir(parents=True, exist_ok=True)

            for fname in FILES:
                in_file = in_dir / fname
                if not in_file.exists():
                    print(f"[SKIP] {project}/{engine}/{fname}: input file not found")
                    continue

                try:
                    data = json.loads(in_file.read_text(encoding="utf-8"))
                except json.JSONDecodeError as e:
                    print(f"[WARN] {project}/{engine}/{fname}: JSON decode error: {e}")
                    continue

                if not isinstance(data, list):
                    print(f"[WARN] {project}/{engine}/{fname}: expected a JSON list, got {type(data)}")
                    continue

                filtered = filter_json_entries(data, allowed)

                out_file = out_dir / fname
                out_file.write_text(json.dumps(filtered, indent=4, ensure_ascii=False), encoding="utf-8")

                print(
                    f"[OK] {project}/{engine}/{fname}: {len(data)} -> {len(filtered)} "
                    f"(allowed packages: {len(allowed)})"
                )
                total_written += 1

    print(f"\nDone. Wrote {total_written} filtered JSON files under: {BASE_OUT}")

if __name__ == "__main__":
    main()
