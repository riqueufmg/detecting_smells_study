from agents.detecting_agent import DetectingAgent

def main():
    project_path = "data/repositories/jsoup"
    processed_path = "data/processed"
    
    detector = DetectingAgent(project_path, processed_path)
    detector.run()
    
if __name__ == "__main__":
    main()