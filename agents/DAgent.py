import util
import os

from langgraph.graph import StateGraph, END
from AgentState import AgentState

class DAgent:

    def __init__(self, project_path):
        self.project_path = project_path
        self.run_path = os.environ.get("RUN_PATH")
        self.agent_state = AgentState()
        self.graph = self._setup_graph()
    
    def init_node(self, state):
        state.messages.append("Agent initialized.")
        self.run_path = util.create_unique_dir()
        return state
    
    def static_analysis_node(self, state):
        try:
            cmd_output = util._run_designite(self.project_path, self.run_path)

            state.analysis_result = cmd_output.stdout
            state.analysis_run = True
            state.messages.append("Designite analysis completed.")

        except Exception as e:
            state.analysis_run = False
            state.messages.append(f"Error occurred during Designite analysis: {e}")

        return state

    def _setup_graph(self):
        graph = StateGraph(AgentState)

        graph.add_node("init", self.init_node)
        graph.add_node("static_analysis", self.static_analysis_node)

        graph.set_entry_point("init")

        graph.add_edge("init", "static_analysis")
        graph.add_edge("static_analysis", END)
        
        return graph.compile()

    def run(self):
        final_state = self.graph.invoke(self.agent_state)
        return final_state