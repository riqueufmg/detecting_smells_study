from dataclasses import dataclass, field

@dataclass
class AgentState:
    messages: list[str] = field(default_factory=list)
    analysis_run: bool = False
    analysis_result: str = ""