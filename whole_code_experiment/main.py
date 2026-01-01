from dotenv import load_dotenv
from pathlib import Path
from utils.clean_projects import ProjectCleaner
from utils.openrouter_engine import OpenRouterEngine
from smells_detection.insufficient_modularization import InsufficientModularizationDetector

def main():
    load_dotenv()

    projects_list = [
        {"project_name": "jsoup"},
        {"project_name": "zxing"},
        {"project_name": "byte-buddy"},
        {"project_name": "google-java-format"},
        {"project_name": "jimfs"},
        {"project_name": "jitwatch"},
    ]

    ## 1. Generate cleaned repositories
    for project in projects_list:
        break ## TODO: remove break after testing
        cleaner = ProjectCleaner(project["project_name"])
        cleaner.clean_repo()
        print(f"Cleaned file path: {cleaned_file}")
    
    for project in projects_list:
        detector = InsufficientModularizationDetector(project["project_name"])

        ## 2. Generate prompts for insufficient modularization detection    
        list_of_prompt_files = detector.generate_prompts()
        print(f"Generated {len(list_of_prompt_files)} prompts for project {project['project_name']}")

        ## 3. Detect insufficient modularization
        detection_count = detector.detect(list_of_prompt_files)
if __name__ == "__main__":
    main()