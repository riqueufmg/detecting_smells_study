class DetectingAgent:
    def __init__(self, project_path, output_path):
        self.project_path = project_path
        self.output_path = output_path

    def collect_metrics(self):
        print("Hello from DetectingAgent!")

    def run(self):
        self.collect_metrics()