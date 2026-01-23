import subprocess
from pathlib import Path

class ArcanRunner:

    def __init__(self, project_path, output_path, classes_path):
        self.project_path = project_path
        self.output_path = output_path
        self.classes_path = classes_path

    def run(self):
        project_name = self.project_path.split("/")[-1]

        Path(self.output_path).mkdir(parents=True, exist_ok=True)

        cmd = [
            "tools/Arcan2/arcan.sh", "analyse",
            "-i", self.project_path,
            "-p", project_name,
            "-o", f"{self.output_path}/arcan",
            "--all", "-l", "JAVA",
        ]
        subprocess.run(cmd, capture_output=True, text=True)
        print(f"Arcan execution completed. Output at {self.output_path}")