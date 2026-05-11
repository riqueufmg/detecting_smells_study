import os

from dotenv import load_dotenv
from DAgent import DAgent

def main():

    project_path = "data/repositories/jsoup" #TODO: change for command args

    load_dotenv()

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise SystemExit("OPENAI_API_KEY not found. Put it in .env (OPENAI_API_KEY=...).")
    
    dagent = DAgent(project_path=project_path)
    state = dagent.run()

    return state

if __name__ == "__main__":
    state = main()

    print(state)