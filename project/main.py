from agents.detecting_agent import DetectingAgent
from pathlib import Path

def main():
    project_path = "data/repositories/jsoup"
    processed_path = "data/processed/metrics/jsoup"
    classes_path = "data/repositories/jsoup/target/classes"
    jar_path = Path("data/repositories/jsoup/target/jsoup-1.22.1-SNAPSHOT.jar")
    
    detector = DetectingAgent(project_path, processed_path, classes_path, jar_path)
    detector.run()
    
if __name__ == "__main__":
    main()