import csv
import random
from pathlib import Path

class FilterPrompts:

    def __init__(self, max_context_size: int = 100_000):
        self.max_context_size = max_context_size

    def save_context_sizes_csv(
        self,
        prompt_files: list[Path],
        output_path: Path
    ):
        rows = []

        for pf in prompt_files:
            with open(pf, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()

            if first_line.startswith("##CONTEXT_SIZE="):
                try:
                    context_size = int(first_line.replace("##CONTEXT_SIZE=", ""))
                except ValueError:
                    context_size = -1
            else:
                context_size = -1

            if context_size <= self.max_context_size:
                rows.append({
                    "prompt_file": str(pf),
                    "context_size": context_size
                })

        output_csv = output_path / "candidates.csv"
        output_csv.parent.mkdir(parents=True, exist_ok=True)

        with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["prompt_file", "context_size"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"[OK] Saved project candidates: {output_csv}")

    def extract_package_and_class_from_prompt(
        self,
        prompt_file: str,
        smell_name: str
    ):
        name = Path(prompt_file).stem
        name = name.replace('_', '.')

        if smell_name in {"God Component", "Unstable Dependency"}:
            return name, ""

        if smell_name in {"Hublike Modularization", "Insufficient Modularization"}:
            parts = name.split(".")
            if len(parts) < 2:
                return name, ""
            return ".".join(parts[:-1]), parts[-1]

        return name, ""

    def build_smell_lookup(self, projects_list: list[str]):
        smell_lookup = {}

        for project in projects_list:
            arch_file = Path(f"data/processed/metrics/{project}/ArchitectureSmells.csv")
            design_file = Path(f"data/processed/metrics/{project}/DesignSmells.csv")

            if arch_file.exists():
                with open(arch_file, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        key = f"{project}::{row['Package']}"
                        smell_lookup.setdefault(key, set()).add(row["Smell"])

            if design_file.exists():
                with open(design_file, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        key = f"{project}::{row['Package']}.{row['Class']}"
                        smell_lookup.setdefault(key, set()).add(row["Smell"])

        return smell_lookup

    def merge_all_candidates(self, projects_list: list[str]):
        smells = {
            "God Component": Path("data/processed/prompts/god_component"),
            "Unstable Dependency": Path("data/processed/prompts/unstable_dependency"),
            "Insufficient Modularization": Path("data/processed/prompts/insufficient_modularization"),
            "Hublike Modularization": Path("data/processed/prompts/hublike_modularization"),
        }

        output_dir = Path("data/processed/candidates")
        output_dir.mkdir(parents=True, exist_ok=True)

        smell_lookup = self.build_smell_lookup(projects_list)

        for smell_name, base_path in smells.items():
            merged_rows = []

            for project in projects_list:
                candidates_csv = base_path / project / "candidates.csv"
                if not candidates_csv.exists():
                    continue

                with open(candidates_csv, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        package, class_name = self.extract_package_and_class_from_prompt(
                            row["prompt_file"],
                            smell_name
                        )

                        if class_name:
                            key = f"{project}::{package}.{class_name}"
                        else:
                            key = f"{project}::{package}"

                        label = 1 if smell_name in smell_lookup.get(key, set()) else 0

                        merged_rows.append({
                            "project": project,
                            "smell": smell_name,
                            "package": package,
                            "class": class_name,
                            "prompt_file": row["prompt_file"],
                            "context_size": row["context_size"],
                            "label": label
                        })

            if not merged_rows:
                print(f"[WARN] No candidates found for smell {smell_name}")
                continue

            output_csv = output_dir / f"{smell_name.replace(' ', '_').lower()}.csv"
            with open(output_csv, "w", newline="", encoding="utf-8") as f:
                fieldnames = [
                    "project",
                    "smell",
                    "package",
                    "class",
                    "prompt_file",
                    "context_size",
                    "label"
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(merged_rows)

            print(f"[OK] Merged candidates with labels saved: {output_csv}")

    def sample_candidates(
        self,
        candidates_csv: Path,
        sample_csv: Path,  # agora é o caminho completo do arquivo de saída
        sample_size: int = 100,
        ratio_positive: float = 0.1
    ):
        """
        Gera um CSV amostrado com sample_size itens a partir do CSV de candidatos,
        mantendo a proporção 90% label 0 e 10% label 1, respeitando disponibilidade.
        """

        if not candidates_csv.exists():
            print(f"[WARN] Candidates CSV não existe: {candidates_csv}")
            return

        sample_csv.parent.mkdir(parents=True, exist_ok=True)

        with open(candidates_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Se não existir coluna label, assume 0 para todos
        for row in rows:
            if "label" not in row:
                row["label"] = int(row.get("label", 0))
            else:
                row["label"] = int(row["label"])

        # Separar por label
        zero_rows = [r for r in rows if r["label"] == 0]
        one_rows = [r for r in rows if r["label"] == 1]

        # Decide quantos pegar
        if len(one_rows) >= int(sample_size * ratio_positive):
            n_one = int(sample_size * ratio_positive)
            n_zero = sample_size - n_one
        else:
            n_one = len(one_rows)
            n_zero = sample_size - n_one

        # Amostra aleatória
        sampled_zeros = random.sample(zero_rows, min(n_zero, len(zero_rows)))
        sampled_ones = random.sample(one_rows, min(n_one, len(one_rows)))
        sampled_rows = sampled_zeros + sampled_ones
        random.shuffle(sampled_rows)

        # Salva CSV no caminho exato recebido em sample_csv
        fieldnames = list(rows[0].keys())
        with open(sample_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sampled_rows)

        print(f"[OK] Sampled CSV saved: {sample_csv}")