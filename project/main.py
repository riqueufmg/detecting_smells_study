from agents.detecting_agent import DetectingAgent
from agents.smells_detection.god_component import GodComponentDetector
from pathlib import Path

def main():
    project_path = "data/repositories/jsoup"
    processed_path = "data/processed/metrics/jsoup"
    classes_path = "data/repositories/jsoup/target/classes"
    jar_path = Path("data/repositories/jsoup/target/jsoup-1.22.1-SNAPSHOT.jar")
    
    detector = DetectingAgent(project_path, processed_path, classes_path, jar_path)
    
    # executa a coleta de métricas
    #metrics_json = detector.collect_metrics()
    print("Metrics collected!")

    smell = {
        "name": "God Component",
        "definition": "when a component is **excessively** large either in terms of Lines Of Code or the number of classes."
    }

    GodComponentDetector(Path(processed_path,"project_metrics.json")).detect(smell)

if __name__ == "__main__":
    main()
