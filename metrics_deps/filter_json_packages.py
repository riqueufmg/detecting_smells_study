import csv
import json
import os
import re
from typing import Any, Dict, List, Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_PATH = os.path.join(BASE_DIR, "gc.csv")
METRICS_BASE = os.path.join(BASE_DIR, "data", "processed", "metrics")
OUT_BASE = os.path.join(BASE_DIR, "data", "filter")

DEFAULT_PACKAGE = "(default package)"

def normalize_pkg_fqn(pkg_fqn: str) -> str:
    pkg_fqn = (pkg_fqn or "").strip()
    return pkg_fqn if pkg_fqn else DEFAULT_PACKAGE

def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())

def make_output_filename(repo_name: str, pkg_fqn: str) -> str:
    return f"{safe_filename(pkg_fqn)}.json"

def load_project_metrics(repo_name: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(METRICS_BASE, repo_name, "project_metrics.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def find_package_blocks_only(project_metrics: Dict[str, Any], pkg_fqn: str) -> List[Dict[str, Any]]:
    matches: List[Dict[str, Any]] = []
    for p in project_metrics.get("packages", []):
        pkg_value = (p.get("package") or p.get("name") or p.get("fqn") or "").strip()
        pkg_value = pkg_value if pkg_value else DEFAULT_PACKAGE
        if pkg_value == pkg_fqn:
            matches.append(p)
    return matches

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def main() -> None:
    print("CSV_PATH:", CSV_PATH)
    print("OUT_BASE:", OUT_BASE)
    print("METRICS_BASE:", METRICS_BASE)

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV não encontrado: {CSV_PATH}")

    ensure_dir(OUT_BASE)

    metrics_cache: Dict[str, Optional[Dict[str, Any]]] = {}
    written = 0

    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")

        for row in reader:
            if not row or len(row) < 3:
                continue

            repo_name = row[0].strip()
            pkg_fqn = normalize_pkg_fqn(row[2])

            if not repo_name:
                continue

            if repo_name not in metrics_cache:
                metrics_cache[repo_name] = load_project_metrics(repo_name)

            project_metrics = metrics_cache[repo_name]

            out_path = os.path.join(OUT_BASE, make_output_filename(repo_name, pkg_fqn))

            if project_metrics is None:
                payload: Any = []
            else:
                matches = find_package_blocks_only(project_metrics, pkg_fqn)
                payload = matches[0] if len(matches) == 1 else matches

            with open(out_path, "w", encoding="utf-8") as out:
                json.dump(payload, out, ensure_ascii=False, indent=2)

            written += 1

    print(f"Concluído. Arquivos gerados: {written}")
    print(f"Saída em: {OUT_BASE}")

if __name__ == "__main__":
    main()
