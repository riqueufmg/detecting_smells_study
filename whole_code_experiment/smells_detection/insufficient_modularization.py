import json
from pathlib import Path
import os
from utils.openrouter_engine import OpenRouterEngine

class InsufficientModularizationDetector:

    def __init__(self, project_name: str):
        self.project_name = project_name
        self.json_path = f"{os.getenv('OUTPUT_PATH')}/metrics/{project_name}/project_metrics.json"
        self.prompts_template_path = Path("data/prompts/templates/detection_insufficient_modularization.tpl")
        self.generated_prompts_dir = Path("data/processed/prompts/insufficient_modularization")

        self.engine = OpenRouterEngine(
            model="meta-llama/llama-3.3-70b-instruct:free",
            max_input_tokens=50000,
            max_output_tokens=2000
        )

    def filter_data(self):
        json_file = Path(self.json_path)
        if not json_file.exists():
            raise FileNotFoundError(f"JSON metrics file not found: {json_file}")

        with open(json_file, "r") as f:
            data = json.load(f)

        classes_list = []

        for pkg in data.get("packages", []):
            for cls in pkg.get("classes", []):
                raw = cls.get("file")
                file_path = Path(raw.replace("data/repositories/", "data/clean_repos/"))

                context_text = ""
                if file_path.exists():
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        context_text = f.read()

                token_count = self.engine.count_tokens(context_text)

                classes_list.append({
                    "package": cls.get("package"),
                    "class": cls.get("class"),
                    "file": cls.get("file"),
                    "cleaned_file": str(file_path),
                    "context_size": token_count,
                })

        return classes_list
    
    def generate_prompts(self) -> list[Path]:
        classes = self.filter_data()

        if not self.prompts_template_path.exists():
            raise FileNotFoundError(f"Template not found: {self.prompts_template_path}")

        with open(self.prompts_template_path, "r", encoding="utf-8") as f:
            template_content = f.read()

        list_of_prompts = []

        project_out_dir = self.generated_prompts_dir / self.project_name
        project_out_dir.mkdir(parents=True, exist_ok=True)

        for cls in classes:
            class_name = cls["class"]
            package_name = cls["package"]
            file_path = cls["file"]
            context_size = cls["context_size"]

            cleaned_file_path = Path(file_path.replace("data/repositories/", "data/clean_repos/"))
            try:
                with open(cleaned_file_path, "r", encoding="utf-8", errors="ignore") as f:
                    code_data = f.read()
            except FileNotFoundError:
                code_data = ""

            prompt_text = template_content \
                .replace("{INPUT_DATA}", code_data) \
                .replace("{SMELL_NAME}", "Insufficient Modularization")

            prompt_text = f"##CONTEXT_SIZE={context_size}\n" + prompt_text

            prompt_file = project_out_dir / f"{package_name}.{class_name}.txt"
            with open(prompt_file, "w", encoding="utf-8") as fp:
                fp.write(prompt_text)

            list_of_prompts.append(prompt_file)

        return list_of_prompts
    
    def detect(self, list_of_prompt_files: list[Path]):
        output_base_dir = Path("data/processed/llm_outputs/insufficient_modularization") / self.project_name
        output_base_dir.mkdir(parents=True, exist_ok=True)

        for prompt_file in list_of_prompt_files:
            with open(prompt_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                context_size = None
                if lines and lines[0].startswith("##CONTEXT_SIZE="):
                    try:
                        context_size = int(lines[0].strip().split("=")[1])
                    except ValueError:
                        context_size = None

                prompt_content = "".join(lines[1:])

                response = self.engine.generate(prompt_content)

                output_file = output_base_dir / f"{prompt_file.stem}.txt"
                with open(output_file, "w", encoding="utf-8") as out_f:
                    out_f.write(response)

                print(f"Saved output to {output_file}")