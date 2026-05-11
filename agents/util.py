import subprocess
import os
import uuid

from pathlib import Path

def _run_designite(project_path, output_path=None):
    
    if output_path is None:
        pass
    else:
        Path(output_path).mkdir(parents=True, exist_ok=True)
        
        cmd = [
                "java", "-jar", "tools/DesigniteJava2.8.3.jar",
                "-i", project_path,
                "-o", output_path,
                #"-c", self.classes_path,
                "-g",
            ]
        
        msg = subprocess.run(cmd, capture_output=True, text=True)
        return msg

def create_unique_dir(base_path="runs"):
    unique_name = str(uuid.uuid4())
    dir_path = os.path.join(base_path, unique_name)
    
    os.makedirs(dir_path, exist_ok=False)
    return dir_path