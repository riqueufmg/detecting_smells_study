import csv
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

CSV_PATH = "hm.csv"
METRICS_BASE = os.path.join("data", "processed", "metrics")
OUT_BASE = os.path.join("data", "filter")

def split_fqn(fqn: str) -> Tuple[str, str]:
    fqn = fqn.strip()
    if "." not in fqn:
        return ("(default package)", fqn)
    pkg, cls = fqn.rsplit(".", 1)
    pkg = pkg.strip() or "(default package)"
    cls = cls.strip()
    return (pkg, cls)

def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())

def make_output_filename(repo_name: str, fqn: str) -> str:
    repo_safe = safe_filename(repo_name)
    fqn_safe = safe_filename(fqn)
    #return f"{repo_safe}__{fqn_safe}.json"
    return f"{fqn_safe}.json"

def load_project_metrics(repo_name: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(METRICS_BASE, repo_name, "project_metrics.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def find_class_blocks_only(project_metrics: Dict[str, Any], pkg: str, cls: str) -> List[Dict[str, Any]]:
    matches: List[Dict[str, Any]] = []
    for p in project_metrics.get("packages", []):
        for c in p.get("classes", []):
            if c.get("package") == pkg and c.get("class") == cls:
                matches.append(c)
    return matches

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def main() -> None:
    ensure_dir(OUT_BASE)

    metrics_cache: Dict[str, Optional[Dict[str, Any]]] = {}

    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        for row in reader:
            if not row or len(row) < 3:
                continue

            repo_name = row[0].strip()
            fqn = row[2].strip()

            if repo_name not in metrics_cache:
                metrics_cache[repo_name] = load_project_metrics(repo_name)

            project_metrics = metrics_cache[repo_name]

            out_filename = make_output_filename(repo_name, fqn)
            out_path = os.path.join(OUT_BASE, out_filename)

            if project_metrics is None:
                payload: Any = []
            else:
                pkg, cls = split_fqn(fqn)
                matches = find_class_blocks_only(project_metrics, pkg, cls)

                if len(matches) == 1:
                    payload = matches[0]
                else:
                    payload = matches

            with open(out_path, "w", encoding="utf-8") as out:
                json.dump(payload, out, ensure_ascii=False, indent=2)

    print(f"Concluído. Todos os arquivos estão em: {OUT_BASE}")

if __name__ == "__main__":
    main()