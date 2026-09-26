from pathlib import Path
from typing import Any

import yaml

from smell_detection.state import DetectionState
from smell_detection.smells_definitions import get_smell_definition, infer_target_type

SUPPORTED_SMELLS = {"IM", "HM", "GC", "UD"}

# function to load configuration attributes
def load_config(state: DetectionState) -> DetectionState:
    config_path = Path(state["config_path"])

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Config file must define a YAML object.")

    return {
        "config": config,
    }


# validate repo path, smell, unit, agent templates, and create output directories
def validate_config(state: DetectionState) -> DetectionState:
    config: dict[str, Any] = state["config"]

    task = config.get("task", {})
    experiment = config.get("experiment", {})
    agent = config.get("agent", {})
    model = config.get("model", {})

    repo_path = task.get("repo_path")
    smell = task.get("smell")
    unit = task.get("unit")

    if not repo_path:
        raise ValueError("Missing task.repo_path in config.")

    if not Path(repo_path).exists():
        raise FileNotFoundError(f"Repository path does not exist: {repo_path}")

    if smell not in SUPPORTED_SMELLS:
        raise ValueError(f"Invalid smell '{smell}'. Expected one of {SUPPORTED_SMELLS}.")

    if not unit:
        raise ValueError("Missing task.unit in config.")

    if not agent.get("system_template"):
        raise ValueError("Missing agent.system_template in config.")

    if not agent.get("instance_template"):
        raise ValueError("Missing agent.instance_template in config.")

    output_dir = experiment.get("output_dir", "data/outputs/verdicts")
    run_dir = experiment.get("run_dir", "data/outputs/runs")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(run_dir).mkdir(parents=True, exist_ok=True)

    return {
        "experiment_name": experiment.get("name", "smell-detection"),
        "output_dir": output_dir,
        "run_dir": run_dir,
        "repo_path": str(Path(repo_path)),
        "smell": smell,
        "unit": unit,
        "model_name": model.get("name", "gpt-5.5"),
        "temperature": float(model.get("temperature", 0.0)),
        "max_verdict_retries": int(agent.get("max_verdict_retries", 1)),
        "attempts": 0,
    }

# update state with target type and smell definition
def prepare_task(state: DetectionState) -> DetectionState:
    smell = state["smell"]

    return {
        "target_type": infer_target_type(smell),
        "smell_definition": get_smell_definition(smell),
    }