import subprocess
from datetime import datetime
from pathlib import Path

from smell_detection.state import DetectionState


def _format_command(command_template: list[str], state: DetectionState) -> list[str]:
    return [
        part.format(
            model=state["model_name"],
            prompt_file=state["prompt_file"],
            repo_path=state["repo_path"],
        )
        for part in command_template
    ]


def _safe_name(value: str) -> str:
    return (
        value.replace("/", "_")
        .replace("\\", "_")
        .replace(".", "_")
        .replace("$", "_")
        .replace(" ", "_")
        .replace("-", "_")
    )


def run_codex(state: DetectionState) -> DetectionState:
    config = state["config"]
    codex_config = config.get("codex", {})

    command_template = codex_config.get("command")
    if not command_template:
        raise ValueError("Missing codex.command in config.")

    command = _format_command(command_template, state)

    cwd_mode = codex_config.get("cwd_mode", "repo")
    cwd = state["repo_path"] if cwd_mode == "repo" else "."

    timeout = int(codex_config.get("timeout", 300))

    run_dir = Path(state["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = _safe_name(state.get("experiment_name", "experiment"))
    smell = state.get("smell", "NA")
    unit = _safe_name(state.get("unit", "unknown"))

    log_prefix = f"{timestamp}_{experiment_name}_{smell}_{unit}"

    command_path = run_dir / f"{log_prefix}_command.txt"
    prompt_path = run_dir / f"{log_prefix}_prompt.md"
    stdout_path = run_dir / f"{log_prefix}_stdout.txt"
    stderr_path = run_dir / f"{log_prefix}_stderr.txt"

    # Mantém também os arquivos last_* para debug rápido
    last_command_path = run_dir / "last_codex_command.txt"
    last_prompt_path = run_dir / "last_codex_prompt.md"
    last_stdout_path = run_dir / "last_codex_stdout.txt"
    last_stderr_path = run_dir / "last_codex_stderr.txt"

    command_text = " ".join(command)

    command_path.write_text(command_text, encoding="utf-8")
    prompt_path.write_text(state["full_prompt"], encoding="utf-8")
    last_command_path.write_text(command_text, encoding="utf-8")
    last_prompt_path.write_text(state["full_prompt"], encoding="utf-8")

    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            input=state["full_prompt"],
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )

        stdout = result.stdout or ""
        stderr = result.stderr or ""
        returncode = result.returncode

    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        returncode = 124

        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")

        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")

        stderr = (
            f"[TIMEOUT] Codex execution exceeded timeout={timeout}s.\n\n"
            f"{stderr}"
        )

    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    last_stdout_path.write_text(stdout, encoding="utf-8")
    last_stderr_path.write_text(stderr, encoding="utf-8")

    return {
        "codex_stdout": stdout,
        "codex_stderr": stderr,
        "codex_returncode": returncode,
        "codex_command_path": str(command_path),
        "codex_prompt_path": str(prompt_path),
        "codex_stdout_path": str(stdout_path),
        "codex_stderr_path": str(stderr_path),
    }