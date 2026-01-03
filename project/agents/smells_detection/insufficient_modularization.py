import json
from pathlib import Path
import os

from agents.llm_inference.llm_engine import LLMInferenceEngine
from agents.llm_inference.gpt_engine import GPTEngine

class InsufficientModularizationDetector:

    def __init__(self, project_name):
        self.project_name = project_name
        self.json_path = f"{os.getenv("OUTPUT_PATH")}/metrics/{project_name}/project_metrics.json"
        self.prompts_path = os.getenv("PROMPTS_PATH")

    def filter_data(self):
        if not Path(self.json_path).exists():
            raise FileNotFoundError(f"JSON file not found: {self.json_path}")

        with open(self.json_path, "r") as f:
            data = json.load(f)

        classes = []

        for package in data.get("packages", []):
            for cls in package.get("classes", []):
                classes.append(cls)

        return classes
    
    def generate_prompts(self, smell):
        classes = self.filter_data()

        with open(
            Path(self.prompts_path, "templates", "detection_insufficient_modularization.tpl"),
            "r"
        ) as file:
            template_content = file.read()

        list_of_prompt_files = []

        for cls in classes:

            prompt = (
                template_content
                .replace("{INPUT_DATA}", json.dumps(cls))
            )

            '''output_dir = Path(
                self.prompts_path,
                "generated",
                "smell_detection",
                smell["smell_name"].replace(" ", "").lower(),
                self.project_name
            )'''

            output_dir = Path(
                os.getenv("OUTPUT_PATH"),
                "prompts",
                "smell_detection",
                "insufficient_modularization",
                self.project_name
            )
            output_dir.mkdir(parents=True, exist_ok=True)

            prompt_file = Path(
                output_dir,
                f"{cls['package'].replace('/', '_')}_{cls['class']}.txt"
            )

            with open(prompt_file, "w") as f:
                f.write(prompt)

            list_of_prompt_files.append(prompt_file)

        return list_of_prompt_files
    
    from pathlib import Path

    def detect_gpt(self, list_of_prompt_files):

        llm_config = {
            "model_name": "gpt-5-mini",
            "max_input_tokens": 100000,
            "max_completion_tokens": 30720
        }

        llm_engine = GPTEngine(**llm_config)

        output_base_dir = Path(
            "data",
            "processed",
            "llm_outputs",
            self.project_name,
            "insufficient_modularization"
        )
        output_base_dir.mkdir(parents=True, exist_ok=True)

        for prompt_file in list_of_prompt_files:

            output_file = output_base_dir / f"{prompt_file.stem}.txt"

            # 1. Skip if already inferred
            if output_file.exists():
                print(f"Output already exists: {output_file.name}")
                continue

            # 2. Load prompt
            with open(prompt_file, "r", encoding="utf-8") as f:
                prompt_content = f.read()

            # 3. Run inference
            response = llm_engine.generate(prompt_content)

            # 4. Safe write (atomic)
            tmp_file = output_file.with_suffix(".tmp")
            with open(tmp_file, "w", encoding="utf-8") as out_f:
                out_f.write(response)

            tmp_file.replace(output_file)

            print(f"Saved output to {output_file.name}")