import json
from pathlib import Path

class GodComponentDetector:

    def __init__(self, json_path: str):
        self.json_path = Path(json_path)

    def get_packages(self) -> list:
        if not self.json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {self.json_path}")

        with open(self.json_path, "r") as f:
            project_json = json.load(f)

        return project_json.get("packages", [])
    
    def detect(self):
        packages = self.get_packages()
        return packages