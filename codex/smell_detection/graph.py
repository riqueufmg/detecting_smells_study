from langgraph.graph import END, StateGraph

from smell_detection.config_loader import (
    load_config,
    prepare_task,
    validate_config,
)
from smell_detection.codex_runner import run_codex
from smell_detection.prompt_render import render_prompts
from smell_detection.state import DetectionState
from smell_detection.veredict_parser import (
    extract_verdict,
    persist_failure,
    persist_result,
    validate_verdict,
)


def increment_attempt(state: DetectionState) -> DetectionState:
    return {
        "attempts": state.get("attempts", 0) + 1
    }

def decide_after_validation(state: DetectionState) -> str:
    if state.get("valid"):
        return "persist_result"

    attempts = state.get("attempts", 0)
    max_retries = state.get("max_verdict_retries", 1)

    if attempts < max_retries:
        return "increment_attempt"

    return "persist_failure"

def build_graph():
    graph = StateGraph(DetectionState)

    # 1) load configuration's attributes
    graph.add_node("load_config", load_config)
    
    # 2) check configuration's attributes
    graph.add_node("validate_config", validate_config)

    # 3) update state for the experiment
    graph.add_node("prepare_task", prepare_task)

    # 4) build prompt with configuration data
    graph.add_node("render_prompts", render_prompts)
    
    graph.add_node("run_codex", run_codex)
    graph.add_node("extract_verdict", extract_verdict)
    graph.add_node("validate_verdict", validate_verdict)
    graph.add_node("increment_attempt", increment_attempt)
    graph.add_node("persist_result", persist_result)
    graph.add_node("persist_failure", persist_failure)

    graph.set_entry_point("load_config")

    graph.add_edge("load_config", "validate_config")
    graph.add_edge("validate_config", "prepare_task")
    graph.add_edge("prepare_task", "render_prompts")
    graph.add_edge("render_prompts", "run_codex")
    graph.add_edge("run_codex", "extract_verdict")
    graph.add_edge("extract_verdict", "validate_verdict")

    graph.add_conditional_edges(
        "validate_verdict",
        decide_after_validation,
        {
            "persist_result": "persist_result",
            "increment_attempt": "increment_attempt",
            "persist_failure": "persist_failure",
        },
    )

    graph.add_edge("increment_attempt", "run_codex")
    graph.add_edge("persist_result", END)
    graph.add_edge("persist_failure", END)

    return graph.compile()