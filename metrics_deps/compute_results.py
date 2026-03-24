import json
from pathlib import Path
from typing import Dict, List, Tuple, Any


class GlobalSmellMetrics:

    def __init__(
        self,
        smell_dir: str,
        engine: str,
        base_path: str = "data/processed/",
        key_field: str = "package",  # "package" ou "identifier"
    ):
        self.base_path = Path(base_path)
        self.smell_dir = smell_dir
        self.engine = engine
        self.key_field = key_field

        self.base_in = self.base_path / "results" / "consolidated_detection"
        self.base_out = self.base_path / "results" / self.smell_dir / self.engine

    @staticmethod
    def classify(pred: bool, gold: bool) -> str:
        if pred and gold:
            return "TP"
        if (not pred) and (not gold):
            return "TN"
        if pred and (not gold):
            return "FP"
        return "FN"

    @staticmethod
    def compute_metrics(cm: Dict[str, int]) -> Dict[str, float]:
        TP, TN, FP, FN = cm["TP"], cm["TN"], cm["FP"], cm["FN"]
        total = TP + TN + FP + FN

        accuracy = (TP + TN) / total if total else 0.0
        precision = TP / (TP + FP) if (TP + FP) else 0.0
        recall = TP / (TP + FN) if (TP + FN) else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

        # Cohen's Kappa
        if total:
            p_observed = (TP + TN) / total

            pred_positive = (TP + FP) / total
            pred_negative = (TN + FN) / total
            gold_positive = (TP + FN) / total
            gold_negative = (TN + FP) / total

            p_expected = (pred_positive * gold_positive) + (pred_negative * gold_negative)

            if (1 - p_expected) == 0:
                kappa = 0.0
            else:
                kappa = (p_observed - p_expected) / (1 - p_expected)
        else:
            kappa = 0.0

        return {
            "accuracy": round(accuracy, 3),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "cohen_kappa": round(kappa, 3),
        }

    def _load_dict(self, path: Path) -> Dict[str, bool]:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON list in {path}, got {type(data)}")

        out: Dict[str, bool] = {}
        for entry in data:
            if not isinstance(entry, dict):
                continue
            detected = bool(entry.get("detection", False))
            key_val: Any = entry.get(self.key_field)

            if isinstance(key_val, list):
                for k in key_val:
                    ks = str(k).strip()
                    if ks:
                        out[ks] = detected
            else:
                ks = "" if key_val is None else str(key_val).strip()
                if ks:
                    out[ks] = detected

        return out

    def _project_dirs(self) -> List[Path]:
        if not self.base_in.exists():
            raise FileNotFoundError(f"Input base not found: {self.base_in}")
        return sorted([p for p in self.base_in.iterdir() if p.is_dir()])

    def _paths_for_project(self, project_dir: Path) -> Tuple[Path, Path]:
        base = project_dir / self.smell_dir / self.engine
        llm_path = base / f"{self.smell_dir}_llm.json"
        designite_path = base / f"{self.smell_dir}_designite.json"
        return llm_path, designite_path

    def generate_global_metrics(self) -> Path:
        cm_global = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}

        used_projects = []
        skipped_projects = []

        total_llm_items = 0
        total_designite_items = 0
        total_union_items = 0

        for project_dir in self._project_dirs():
            llm_path, designite_path = self._paths_for_project(project_dir)

            if not llm_path.exists() or not designite_path.exists():
                skipped_projects.append(project_dir.name)
                continue

            llm_dict = self._load_dict(llm_path)
            designite_dict = self._load_dict(designite_path)

            keys = set(llm_dict.keys()) | set(designite_dict.keys())

            for k in keys:
                pred = llm_dict.get(k, False)
                gold = designite_dict.get(k, False)
                cm_global[self.classify(pred, gold)] += 1

            used_projects.append(project_dir.name)
            total_llm_items += len(llm_dict)
            total_designite_items += len(designite_dict)
            total_union_items += len(keys)

        metrics = self.compute_metrics(cm_global)

        self.base_out.mkdir(parents=True, exist_ok=True)
        out_path = self.base_out / f"{self.smell_dir}_metrics_global.json"

        payload = {
            "scope": "global_across_projects",
            "smell": self.smell_dir,
            "engine": self.engine,
            "key_field": self.key_field,
            "confusion_matrix": cm_global,
            "metrics": metrics,
            "counts": {
                "projects_used": len(used_projects),
                "projects_skipped": len(skipped_projects),
                "llm_items_sum": total_llm_items,
                "designite_items_sum": total_designite_items,
                "union_items_sum": total_union_items,
                "classified_items": sum(cm_global.values()),
            },
            "projects": {
                "used": used_projects,
                "skipped": skipped_projects,
            },
            "paths": {
                "inputs_base": str(self.base_in),
                "output": str(out_path),
            },
        }

        out_path.write_text(json.dumps(payload, indent=4, ensure_ascii=False), encoding="utf-8")
        return out_path


if __name__ == "__main__":
    ENGINES = ["gpt", "deepseek", "qwen"]

    package_smells = ["god_component", "unstable_dependency", "hublike_modularization"]
    for smell in package_smells:
        for engine in ENGINES:
            gen = GlobalSmellMetrics(
                smell_dir=smell,
                engine=engine,
                base_path="data/processed/",
                key_field="package",
            )
            try:
                out = gen.generate_global_metrics()
                print(f"[OK] wrote {out}")
            except FileNotFoundError as e:
                print(f"[ERR] {e}")

    for engine in ENGINES:
        gen = GlobalSmellMetrics(
            smell_dir="insufficient_modularization",
            engine=engine,
            base_path="data/processed/",
            key_field="identifier",
        )
        try:
            out = gen.generate_global_metrics()
            print(f"[OK] wrote {out}")
        except FileNotFoundError as e:
            print(f"[ERR] {e}")