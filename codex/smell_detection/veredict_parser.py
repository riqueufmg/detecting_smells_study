import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from smell_detection.state import DetectionState


REQUIRED_KEYS = {"target", "detection", "justification"}


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in Codex output.")

    return json.loads(match.group(0))

def extract_verdict(state: DetectionState) -> DetectionState:
    stdout = state.get("codex_stdout", "")
    stderr = state.get("codex_stderr", "")

    try:
        verdict = _extract_json_object(stdout)
    except Exception as exc:
        return {
            "verdict": {},
            "valid": False,
            "error": (
                f"Could not extract verdict JSON from Codex stdout. "
                f"Reason: {exc}. "
                f"Codex return code: {state.get('codex_returncode')}. "
                f"Stderr preview: {stderr[:1000] if stderr else '<empty stderr>'}"
            ),
        }

    return {
        "verdict": verdict,
    }

def validate_verdict(state: DetectionState) -> DetectionState:
    if state.get("valid") is False and state.get("error"):
        return state

    verdict = state.get("verdict")

    if not isinstance(verdict, dict):
        return {
            "valid": False,
            "error": "Verdict is not a JSON object.",
        }

    keys = set(verdict.keys())
    if keys != REQUIRED_KEYS:
        return {
            "valid": False,
            "error": f"Invalid verdict keys. Expected {REQUIRED_KEYS}, got {keys}.",
        }

    if verdict["target"] != state["unit"]:
        return {
            "valid": False,
            "error": f"Target mismatch. Expected {state['unit']}, got {verdict['target']}.",
        }

    if not isinstance(verdict["detection"], bool):
        return {
            "valid": False,
            "error": "Field detection must be a JSON boolean.",
        }

    if not isinstance(verdict["justification"], str) or not verdict["justification"].strip():
        return {
            "valid": False,
            "error": "Field justification must be a non-empty string.",
        }

    return {
        "valid": True,
        "error": None,
    }

def persist_failure(state: DetectionState) -> DetectionState:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_unit = state["unit"].replace(".", "_").replace("/", "_")

    failure_path = (
        Path(state["output_dir"])
        / f"{timestamp}_{state['smell']}_{safe_unit}_failure.json"
    )

    failure_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "target": state.get("unit"),
        "detection": None,
        "justification": None,
        "valid": False,
        "error": state.get("error"),
        "codex_returncode": state.get("codex_returncode"),
        "codex_command_path": state.get("codex_command_path"),
        "codex_prompt_path": state.get("codex_prompt_path"),
        "codex_stdout_path": state.get("codex_stdout_path"),
        "codex_stderr_path": state.get("codex_stderr_path"),
        "codex_stdout_preview": (state.get("codex_stdout") or "")[:2000],
        "codex_stderr_preview": (state.get("codex_stderr") or "")[:2000],
    }

    failure_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "verdict_path": str(failure_path),
    }

def persist_result(state: DetectionState) -> DetectionState:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_unit = state["unit"].replace(".", "_").replace("/", "_")

    output_dir = Path(state["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    verdict_path = (
        output_dir
        / f"{timestamp}_{state['smell']}_{safe_unit}_detection.json"
    )

    payload = {
        "target": state["verdict"]["target"],
        "detection": state["verdict"]["detection"],
        "justification": state["verdict"]["justification"],
    }

    verdict_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "verdict_path": str(verdict_path),
    }