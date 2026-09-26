"""SmellAgent: a thin DefaultAgent subclass that validates the submitted verdict.

Mini-swe-agent's submission path: the agent runs
`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat verdict.json`; the environment
raises `Submitted` with `submission` == the verdict.json contents; `DefaultAgent.run`
then appends a `role:"exit"` message and stops.

This subclass intercepts that `Submitted`, parses + schema-validates the JSON.
On a valid verdict it exits normally (enriching the exit message with the parsed
verdict so scoring is trivial). On a malformed verdict it bounces a correction
message back to the model up to `max_verdict_retries` times; if still malformed,
it exits with status `MalformedVerdict` (recorded, never silently dropped — this
matches the pre-registered exclusion rule). No other agent behaviour changes:
the linear history is preserved exactly as in upstream mini-swe-agent.
"""

import json
import re

from minisweagent.agents.default import AgentConfig, DefaultAgent
from minisweagent.exceptions import Submitted

# Reused so the static and agentic arms share one scorer.
_REQUIRED_KEYS = {"class", "detection", "justification"}
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

# Provenance copied from extra_template_vars into info.provenance so every
# trajectory is self-describing (the smell definition is embedded as plain
# text — no hashing).
_PROVENANCE_KEYS = (
    "cell_id", "project", "smell", "unit", "unit_kind", "unit_id",
    "model_slug", "repo_host_path",
    "expected_commit_sha", "observed_commit_sha", "repetition",
    "designite", "smell_definition",
)


class SmellAgentConfig(AgentConfig):
    max_verdict_retries: int = 1
    """How many malformed verdict.json submissions may be bounced back for
    correction before the run is recorded as MalformedVerdict."""


def parse_verdict(submission: str) -> tuple[dict | None, str]:
    """Return (verdict, "") if `submission` is a schema-valid verdict, else
    (None, <reason>). The schema is {class:str, detection:bool, justification:str}.
    """
    match = _JSON_OBJECT_RE.search(submission or "")
    if not match:
        return None, "verdict.json did not contain a JSON object."
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        return None, f"verdict.json is not valid JSON: {e}."
    if not isinstance(obj, dict):
        return None, "verdict.json must be a JSON object."
    missing = _REQUIRED_KEYS - obj.keys()
    if missing:
        return None, f"verdict.json is missing required key(s): {sorted(missing)}."
    extra = obj.keys() - _REQUIRED_KEYS
    if extra:
        return None, f"verdict.json has unexpected key(s): {sorted(extra)}."
    if not isinstance(obj["detection"], bool):
        return None, "'detection' must be a JSON boolean (true or false)."
    if not isinstance(obj["class"], str) or not isinstance(obj["justification"], str):
        return None, "'class' and 'justification' must be strings."
    return obj, ""


class SmellAgent(DefaultAgent):
    """DefaultAgent that enforces the verdict schema (see module docstring)."""

    def __init__(self, model, env, **kwargs):
        super().__init__(model, env, config_class=SmellAgentConfig, **kwargs)
        self._verdict_retries_used = 0

    def serialize(self, *extra_dicts) -> dict:
        """Embed cell provenance (incl. the plain-text smell definition and
        the expected/observed target commit) into the saved trajectory."""
        provenance = {k: self.extra_template_vars.get(k) for k in _PROVENANCE_KEYS}
        return super().serialize({"info": {"provenance": provenance}}, *extra_dicts)

    def step(self) -> list[dict]:
        try:
            return super().step()
        except Submitted as submitted:
            exit_msg = submitted.messages[0]
            raw = exit_msg.get("extra", {}).get("submission", "")
            verdict, reason = parse_verdict(raw)

            if verdict is not None:
                # Valid: exit normally, but enrich the exit message so the
                # post-hoc scorer can read messages[-1].extra.verdict directly.
                exit_msg["extra"]["verdict"] = verdict
                exit_msg["extra"]["verdict_valid"] = True
                raise Submitted(exit_msg) from None

            if self._verdict_retries_used < self.config.max_verdict_retries:
                # Malformed but retry budget remains: keep the assistant's
                # submit message in history (already appended by query()) and
                # append a user correction message so the loop continues.
                self._verdict_retries_used += 1
                # The env raised Submitted mid-step, before execute_actions
                # could call format_observation_messages, so the assistant's
                # tool_calls are still unanswered. OpenAI rejects the next
                # turn unless every tool_call_id has a paired role:"tool"
                # reply (other providers tolerate the gap); synthesize them.
                self._answer_pending_tool_calls(raw)
                correction = (
                    f"Your submission was rejected: {reason}\n\n"
                    f"Rewrite verdict.json so it is EXACTLY one JSON object with keys "
                    f'"class", "detection" (boolean), "justification" (string) and '
                    f"nothing else, then resubmit with the single command:\n"
                    f"echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat verdict.json"
                )
                return self.add_messages(self.model.format_message(role="user", content=correction))

            # Out of retries: record (do not drop) as MalformedVerdict.
            exit_msg["extra"]["exit_status"] = "MalformedVerdict"
            exit_msg["extra"]["verdict"] = None
            exit_msg["extra"]["verdict_valid"] = False
            exit_msg["extra"]["verdict_error"] = reason
            raise Submitted(exit_msg) from None

    def _answer_pending_tool_calls(self, submission: str) -> None:
        """Append a synthetic role:"tool" reply for every tool_call_id in
        the last assistant message. Used only on the verdict-retry path,
        where Submitted was raised before format_observation_messages had
        a chance to run."""
        if not self.messages:
            return
        last = self.messages[-1]
        actions = (last.get("extra") or {}).get("actions") or []
        if not actions:
            return
        synthetic = [
            {"output": submission, "returncode": 0, "exception_info": None}
            for _ in actions
        ]
        tool_msgs = self.model.format_observation_messages(
            last, synthetic, self.get_template_vars()
        )
        self.add_messages(*tool_msgs)
