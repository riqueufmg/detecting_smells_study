# file: multi_agent_smell_detector_single_rule.py
"""
Multi-agent smell detector with:
- ONE single rule (rubric) generated once per run (per smell) and reused for all samples
- loops:
  - invalid JSON -> repair (max 3) then finalize with last valid output
  - auditor disagrees -> retry judge (max 3) then finalize with last result

Output schema (baseline-compatible):
  Package-level smells: {"package": "...", "detection": bool, "justification": "..."}
  Class-level smells:   {"class":   "...", "detection": bool, "justification": "..."}
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Literal, Optional, TypedDict

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

SmellType = Literal[
    "unstable_dependency",
    "god_component",
    "insufficient_modularization",
    "hublike_modularization",
]

Level = Literal["package", "class"]

MAX_INVALID_JSON_RETRIES = 3
MAX_AUDIT_DISAGREE_RETRIES = 3


class State(TypedDict, total=False):
    # inputs
    json_path: str
    smell_type: SmellType
    smell_definition: str

    # shared rubric (single rule)
    rule_spec: Dict[str, Any]

    # loaded sample
    sample_raw: Dict[str, Any]
    entity_level: Level
    entity_key: str  # "package" | "class"
    entity_name: str

    # judge I/O
    judge_prompt: str
    llm_raw_text: str
    llm_raw_json: Dict[str, Any]

    # output (baseline)
    output_json: Dict[str, Any]
    last_output_json: Dict[str, Any]

    # repair control
    needs_repair: bool
    invalid_json_retries: int

    # audit
    audit_prompt: str
    audit_raw_text: str
    audit_ok: bool
    audit_reason: str
    audit_disagree_retries: int


# ----------------------------
# Helpers
# ----------------------------
def infer_level(smell_type: SmellType) -> Level:
    return "package" if smell_type in ("unstable_dependency", "god_component") else "class"


def entity_key_for_level(level: Level) -> str:
    return "package" if level == "package" else "class"


def load_json(path: str) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    t = (text or "").strip()

    # Remove fenced blocks if present
    if t.startswith("```"):
        first_nl = t.find("\n")
        if first_nl != -1:
            t2 = t[first_nl + 1 :]
            end = t2.rfind("```")
            if end != -1:
                t = t2[:end].strip()

    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def validate_to_baseline(entity_key: str, entity_name: str, llm_obj: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    val = llm_obj.get(entity_key, entity_name)
    if not isinstance(val, str) or not val.strip():
        val = entity_name
    out[entity_key] = val

    det = llm_obj.get("detection", False)
    if isinstance(det, bool):
        out["detection"] = det
    elif isinstance(det, str):
        out["detection"] = det.strip().lower() in ("true", "yes", "1")
    else:
        out["detection"] = False

    just = llm_obj.get("justification", "")
    if not isinstance(just, str):
        just = str(just)
    out["justification"] = just

    return out


def shorten_text(s: str, max_len: int = 12000) -> str:
    return (s or "")[:max_len]


# ----------------------------
# Prompt builders
# ----------------------------
def build_rule_prompt(smell_type: SmellType, smell_definition: str, level: Level) -> str:
    return f"""You are an expert in architecture smell detection.

Smell: {smell_type}
Level: {level}
Definition: {smell_definition}

Create an OPERATIONAL decision rubric as JSON.
Requirements:
- Output MUST be valid JSON (no markdown).
- The rubric must be applicable to ANY instance (not tailored to one sample).
- Each criterion must include: id, description, how_to_check (explicit), weight (1-5).
- Include a decision_policy and margins.

Output schema:
{{
  "smell_type": "{smell_type}",
  "level": "{level}",
  "criteria": [
    {{
      "id": "C1",
      "description": "...",
      "how_to_check": "...",
      "weight": 3
    }}
  ],
  "decision_policy": {{
    "type": "score_threshold" | "boolean",
    "threshold": 5,
    "notes": "..."
  }},
  "margins": {{
    "instability_delta": 0.05,
    "notes": "..."
  }}
}}
"""


def build_judge_prompt(
    sample_raw: Dict[str, Any],
    smell_type: SmellType,
    smell_definition: str,
    rule_spec: Dict[str, Any],
    level: Level,
    entity_key: str,
    entity_name: str,
    audit_reason: Optional[str],
    invalid_json_retries: int,
    audit_disagree_retries: int,
) -> str:
    feedback = ""
    if audit_reason:
        feedback = f"\n# Auditor feedback from previous attempt\n{audit_reason}\n"

    return f"""# Task
You are a software engineering expert in software refactoring.
Detect the smell: {smell_type}.
Definition: {smell_definition}

# Rubric (MUST FOLLOW)
```json
{json.dumps(rule_spec, ensure_ascii=False)}
```
{feedback}
# Constraints
- Analyze at the **{level} level** and evaluate this {entity_key} individually.
- Apply the rubric strictly.
- Output MUST be valid JSON and MUST match exactly the schema in #Output (NO extra keys).
- Do NOT include any text outside the JSON.
- Attempt info: invalid_json_retries={invalid_json_retries}, audit_disagree_retries={audit_disagree_retries}

# Input
```json
{json.dumps(sample_raw, ensure_ascii=False)}
```

# Output
```json
{{
  "{entity_key}": "{entity_name}",
  "detection": false,
  "justification": "Cite which rubric criteria were met/not met, with concrete values from the input"
}}
```
"""


def build_audit_prompt(
    baseline_output: Dict[str, Any],
    rule_spec: Dict[str, Any],
    sample_raw: Dict[str, Any],
) -> str:
    return f"""You are an auditor. Verify whether the decision and justification follow the rubric and are supported by the sample values.

Rubric:
{json.dumps(rule_spec, ensure_ascii=False)}

Sample input:
{json.dumps(sample_raw, ensure_ascii=False)}

Model output:
{json.dumps(baseline_output, ensure_ascii=False)}

Return ONLY valid JSON:
{{"audit_ok": true, "reason": "..."}}"""


# ----------------------------
# Agents (nodes) — detection graph (rule_spec is injected)
# ----------------------------
def init_agent(state: State) -> State:
    state.setdefault("invalid_json_retries", 0)
    state.setdefault("audit_disagree_retries", 0)
    state.setdefault("audit_reason", "")
    state.setdefault("last_output_json", {})
    return state


def loader_agent(state: State) -> State:
    raw = load_json(state["json_path"])

    level = infer_level(state["smell_type"])
    ekey = entity_key_for_level(level)

    if ekey == "package":
        ename = str(raw.get("package", "")).strip() or "UNKNOWN_PACKAGE"
    else:
        ename = str(raw.get("class", "")).strip() or str(raw.get("entity", "")).strip() or "UNKNOWN_CLASS"

    state["sample_raw"] = raw
    state["entity_level"] = level
    state["entity_key"] = ekey
    state["entity_name"] = ename
    return state


def judge_agent(state: State) -> State:
    prompt = build_judge_prompt(
        sample_raw=state["sample_raw"],
        smell_type=state["smell_type"],
        smell_definition=state["smell_definition"],
        rule_spec=state["rule_spec"],
        level=state["entity_level"],
        entity_key=state["entity_key"],
        entity_name=state["entity_name"],
        audit_reason=(state.get("audit_reason") or None),
        invalid_json_retries=state.get("invalid_json_retries", 0),
        audit_disagree_retries=state.get("audit_disagree_retries", 0),
    )
    state["judge_prompt"] = prompt

    llm = ChatOpenAI(model="gpt-5-mini", temperature=0)
    chain = llm | StrOutputParser()
    messages = [
        SystemMessage(content="Return ONLY valid JSON matching the schema. No extra keys, no extra text."),
        HumanMessage(content=prompt),
    ]
    state["llm_raw_text"] = chain.invoke(messages)
    return state


def validator_agent(state: State) -> State:
    obj = safe_json_loads(state.get("llm_raw_text", ""))
    if obj is None:
        state["needs_repair"] = True
        state["llm_raw_json"] = {}
        return state

    state["needs_repair"] = False
    state["llm_raw_json"] = obj

    out = validate_to_baseline(state["entity_key"], state["entity_name"], obj)
    state["output_json"] = out
    state["last_output_json"] = out
    return state


def repair_agent(state: State) -> State:
    llm = ChatOpenAI(model="gpt-5-mini", temperature=0)
    chain = llm | StrOutputParser()

    entity_key = state["entity_key"]
    repair_prompt = f"""Your previous answer was not valid JSON.

Fix it and return ONLY valid JSON in exactly this schema (no extra keys):
{{
  "{entity_key}": "{state["entity_name"]}",
  "detection": false,
  "justification": "..."
}}

Previous answer:
{shorten_text(state.get("llm_raw_text", ""))}
"""
    messages = [
        SystemMessage(content="Return ONLY valid JSON. No markdown fences."),
        HumanMessage(content=repair_prompt),
    ]
    state["llm_raw_text"] = chain.invoke(messages)
    return state


def auditor_agent(state: State) -> State:
    audit_prompt = build_audit_prompt(
        baseline_output=state["output_json"],
        rule_spec=state["rule_spec"],
        sample_raw=state["sample_raw"],
    )
    state["audit_prompt"] = audit_prompt

    llm = ChatOpenAI(model="gpt-5-mini", temperature=0)
    chain = llm | StrOutputParser()
    messages = [
        SystemMessage(content="Return ONLY valid JSON."),
        HumanMessage(content=audit_prompt),
    ]
    text = chain.invoke(messages)
    state["audit_raw_text"] = text

    obj = safe_json_loads(text) or {}
    audit_ok = bool(obj.get("audit_ok", True))
    reason = obj.get("reason", "")
    if not isinstance(reason, str):
        reason = str(reason)

    state["audit_ok"] = audit_ok
    state["audit_reason"] = reason
    return state


def finalize_agent(state: State) -> State:
    if state.get("output_json"):
        return state
    last = state.get("last_output_json") or {}
    if last:
        state["output_json"] = last
        return state

    entity_key = state.get("entity_key") or "package"
    entity_name = state.get("entity_name") or "UNKNOWN"
    state["output_json"] = {
        entity_key: entity_name,
        "detection": False,
        "justification": "No valid model output produced.",
    }
    return state


# ----------------------------
# Routing
# ----------------------------
def route_after_validate(state: State) -> str:
    if state.get("needs_repair"):
        retries = state.get("invalid_json_retries", 0)
        if retries < MAX_INVALID_JSON_RETRIES:
            state["invalid_json_retries"] = retries + 1
            return "repair"
        return "finalize"
    return "audit"


def route_after_audit(state: State) -> str:
    if state.get("audit_ok") is False:
        tries = state.get("audit_disagree_retries", 0)
        if tries < MAX_AUDIT_DISAGREE_RETRIES:
            state["audit_disagree_retries"] = tries + 1
            return "judge"
        return "finalize"
    return "finalize"


# ----------------------------
# Graph builders
# ----------------------------
def build_detection_graph():
    """
    Graph expects rule_spec to already be present in the initial state.
    """
    g = StateGraph(State)

    g.add_node("init", init_agent)
    g.add_node("loader", loader_agent)
    g.add_node("judge", judge_agent)
    g.add_node("validator", validator_agent)
    g.add_node("repair", repair_agent)
    g.add_node("audit", auditor_agent)
    g.add_node("finalize", finalize_agent)

    g.set_entry_point("init")

    g.add_edge("init", "loader")
    g.add_edge("loader", "judge")
    g.add_edge("judge", "validator")

    g.add_conditional_edges(
        "validator",
        route_after_validate,
        {"repair": "repair", "audit": "audit", "finalize": "finalize"},
    )

    g.add_edge("repair", "validator")

    g.add_conditional_edges(
        "audit",
        route_after_audit,
        {"judge": "judge", "finalize": "finalize"},
    )

    g.add_edge("finalize", END)

    return g.compile()


def build_single_rule(
    smell_type: SmellType,
    smell_definition: str,
    model: str = "gpt-5-mini",
) -> Dict[str, Any]:
    """
    Generate ONE rubric (rule_spec) using only smell_type + definition.
    """
    level = infer_level(smell_type)
    prompt = build_rule_prompt(smell_type, smell_definition, level)

    llm = ChatOpenAI(model=model, temperature=0)
    chain = llm | StrOutputParser()
    messages = [
        SystemMessage(content="Return ONLY valid JSON. No markdown."),
        HumanMessage(content=prompt),
    ]
    text = chain.invoke(messages)
    obj = safe_json_loads(text)
    if obj is None:
        # fallback rubric
        obj = {
            "smell_type": smell_type,
            "level": level,
            "criteria": [],
            "decision_policy": {"type": "boolean", "threshold": 0, "notes": "Fallback empty rubric."},
            "margins": {"instability_delta": 0.05, "notes": "Fallback."},
        }
    return obj


def save_rule(rule_spec: Dict[str, Any], path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rule_spec, ensure_ascii=False, indent=2), encoding="utf-8")


def load_rule(path: str) -> Dict[str, Any]:
    return load_json(path)


# ----------------------------
# Public API (single rule)
# ----------------------------
def run_one_with_rule(
    json_path: str,
    smell_type: SmellType,
    smell_definition: str,
    rule_spec: Dict[str, Any],
) -> Dict[str, Any]:
    app = build_detection_graph()
    st = app.invoke(
        {
            "json_path": json_path,
            "smell_type": smell_type,
            "smell_definition": smell_definition,
            "rule_spec": rule_spec,  # injected single rule
        }
    )
    return st["output_json"]


def run_folder_with_single_rule(
    root_dir: str,
    smell_type: SmellType,
    smell_definition: str,
    out_path: str,
    rule_path: Optional[str] = None,
    model: str = "gpt-5-mini",
) -> None:
    """
    Generates ONE rule, then applies it to all JSON files in root_dir,
    writing JSONL outputs to out_path.

    If rule_path is provided:
      - if exists: loads and uses it
      - if not exists: generates rule and saves it
    """
    root = Path(root_dir)
    json_files = sorted(root.rglob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No .json files found in: {root_dir}")

    # load or build rule once
    if rule_path and Path(rule_path).exists():
        rule_spec = load_rule(rule_path)
    else:
        rule_spec = build_single_rule(smell_type, smell_definition, model=model)
        if rule_path:
            save_rule(rule_spec, rule_path)

    app = build_detection_graph()

    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    with out_p.open("w", encoding="utf-8") as out:
        for p in json_files:
            st = app.invoke(
                {
                    "json_path": str(p),
                    "smell_type": smell_type,
                    "smell_definition": smell_definition,
                    "rule_spec": rule_spec,
                }
            )
            out.write(json.dumps(st["output_json"], ensure_ascii=False) + "\n")


if __name__ == "__main__":
    print("multi_agent_smell_detector_single_rule.py loaded.")
