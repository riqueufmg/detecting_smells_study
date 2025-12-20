import json
from pathlib import Path

class GodComponentDetector:

    def __init__(self, json_path):
        self.json_path = Path(json_path)

    def load_packages(self):
        if not self.json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {self.json_path}")

        with open(self.json_path, "r") as f:
            data = json.load(f)

        return data.get("packages", [])
    
    def detect(self, smell):
        packages = self.load_packages()

        detects = dict()

        for package in packages:
            with open("agents/prompts/smells_detection.tpl", "r") as file:
                template_content = file.read()
            
            data = {
                "SMELL_NAME": smell['name'],
                "SMELL_DEFINITION": smell['definition'],
                "INPUT_DATA": json.dumps(package)
            }

            prompt = template_content.replace("{SMELL_NAME}", smell['name']) \
                         .replace("{SMELL_DEFINITION}", smell['definition']) \
                         .replace("{INPUT_DATA}", json.dumps(package))


            print(prompt)

            break

        return 0