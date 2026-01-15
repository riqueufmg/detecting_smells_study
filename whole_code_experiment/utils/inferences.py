import json
import os
from pathlib import Path
from utils.openrouter_engine import OpenRouterEngine
from utils.gpt_engine import GPTEngine

class Inference:

    def __init__(self, smell):
        self.engine = OpenRouterEngine(model="gpt-5-mini")
        self.smell = smell

    def detect_gpt(self, list_of_prompt_files):

        llm_config = {
            "model_name": "gpt-5-mini",
            "max_input_tokens": 100000,
            "max_completion_tokens": 30720,
        }

        llm_engine = GPTEngine(**llm_config)

        for prompt_file in list_of_prompt_files:

            with open(prompt_file, "r") as f:
                prompt_content = f.read()
            
            output_dir = Path(
                os.getenv("OUTPUT_PATH"),
                "llm_outputs",
                self.smell,
                "gpt"
            )
            output_dir.mkdir(parents=True, exist_ok=True)

            response = llm_engine.generate(prompt_content)

            output_file = output_dir / f"{prompt_file.stem}.txt"
            
            with open(output_file, "w") as out_f:
                out_f.write(response)