import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents.structured_output import ToolStrategy
from langchain.agents import create_agent
from dataclasses import dataclass
from typing import List

PLANNER_PROMPT = """
God Component is a software design where a Java package is very large and has many responsibilities.

Define a set of rules to detect God Component in package level.

Constraints:
- Consider size, coupling, cohesion and complexity.
- Rules must be verifiable.
- Define between 3 and 5 rules.
"""

@dataclass
class ResponseFormat:
    role_name: str
    verification: str
    weight: str | None = None

class DAgent:
    def __init__(self, api_key: str, model_name: str):
        self.model = init_chat_model(
            model_name,
            temperature=0.1,
        )

        self.planner_agent = create_agent(
            model=self.model,
            system_prompt=PLANNER_PROMPT,
            response_format=ToolStrategy(ResponseFormat),
        )
    
    def run_planner_agent(self):

        response = self.planner_agent.invoke({
            "messages": [
                {
                    "role": "user",
                    "content": "Generate the rules."
                }
            ]
        })

        return response["structured_response"]

def main():
    load_dotenv()

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise SystemExit(
            "OPENAI_API_KEY not found. Put it in .env (OPENAI_API_KEY=...) or export it in your shell." 
        )

    agents = DAgent(OPENAI_API_KEY, "gpt-5-mini")

    response = agents.run_planner_agent()

    print(response)

if __name__ == "__main__":
    main()