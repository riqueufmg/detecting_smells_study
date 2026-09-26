from typing import Any, Literal, Optional, TypedDict


SmellType = Literal["IM", "HM", "GC", "UD"]
UnitKind = Literal["Java class", "Java package"]


class DetectionState(TypedDict, total=False):
    config_path: str
    config: dict[str, Any]

    experiment_name: str
    output_dir: str
    run_dir: str

    repo_path: str
    smell: SmellType
    unit: str
    target_type: UnitKind
    smell_definition: str

    model_name: str
    temperature: float

    system_prompt: str
    instance_prompt: str
    full_prompt: str
    prompt_file: str

    codex_stdout: str
    codex_stderr: str
    codex_returncode: int

    verdict_path: str
    verdict: dict[str, Any]

    attempts: int
    max_verdict_retries: int

    valid: bool
    error: Optional[str]