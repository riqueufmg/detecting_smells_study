from datetime import datetime
from pathlib import Path

from jinja2 import Template

from smell_detection.state import DetectionState

def render_prompts(state: DetectionState) -> DetectionState:
    config = state["config"]
    agent = config["agent"]

    variables = {
        "repo_path": state["repo_path"],
        "smell": state["smell"],
        "unit": state["unit"],
        "target_type": state["target_type"],
        "smell_definition": state["smell_definition"],
        "model": state["model_name"],
    }

    system_prompt = Template(agent["system_template"]).render(**variables)
    instance_prompt = Template(agent["instance_template"]).render(**variables)

    full_prompt = (
        "# System instructions\n\n"
        f"{system_prompt}\n\n"
        "# Instance instructions\n\n"
        f"{instance_prompt}\n"
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_unit = state["unit"].replace(".", "_").replace("/", "_")
    prompt_file = Path(state["run_dir"]) / f"{timestamp}_{state['smell']}_{safe_unit}_prompt.md"

    prompt_file.parent.mkdir(parents=True, exist_ok=True)
    prompt_file.write_text(full_prompt, encoding="utf-8")

    return {
        "system_prompt": system_prompt,
        "instance_prompt": instance_prompt,
        "full_prompt": full_prompt,
        "prompt_file": str(prompt_file),
    }