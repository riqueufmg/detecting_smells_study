"""Canonical smell registry shared by the static and agentic arms.

`SMELL_DEFINITIONS` is byte-identical to the `Definition:` block of the
static-arm prompt templates in
`raw_code/data/prompts/templates/detection_<smell>.tpl`. run_batch.py
re-verifies this at launch (assert_definitions_match_templates) so the two
arms can never silently drift.

`SMELL_META` records, per smell, the pre-registered experimental matrix:
  - display : human name as written in the template
  - xlsx    : the candidate workbook under samples/ (first sheet is read)
  - level   : the granularity evaluated by the static arm:
                god_component, unstable_dependency        -> "package"
                hublike_modularization,
                insufficient_modularization               -> "class"

For class-level smells the workbook carries an explicit `fqn` column which
is used verbatim (it handles nested classes correctly, where `package`
already includes the outer class).
"""

from __future__ import annotations

SMELL_DEFINITIONS: dict[str, str] = {
    "god_component": "God Component arise when a component is **excessively** large either in terms of Lines Of Code or the number of classes.",
    "hublike_modularization": "Hub-like Modularization  arises when an abstraction has dependencies (both incoming and outgoing) with a large number of other abstractions.",
    "insufficient_modularization": "Insufficient Modularization arises when a class represents an abstraction that has not been adequately decomposed, resulting in excessive size, complexity, or a bloated interface. Such classes are difficult to understand, maintain, and evolve, and often concentrate responsibilities that could be separated into smaller, more cohesive abstractions.",
    "unstable_dependency": "Unstable Dependency arises when a component depends on other less stable components. Stable Dependencies Principle states that the dependencies between packages should be in the direction of the stability of packages. Hence, a package should only depend on packages that are more stable than itself. An unstable dependency architecture smell occurs when this principle is not followed.",
}

SMELL_META: dict[str, dict[str, str]] = {
    "god_component": {
        "display": "God Component",
        "xlsx": "metrics_gc.xlsx",
        "level": "package",
    },
    "hublike_modularization": {
        "display": "Hub-like Modularization",
        "xlsx": "metrics_hm.xlsx",
        "level": "class",
    },
    "insufficient_modularization": {
        "display": "Insufficient Modularization",
        "xlsx": "metrics_im.xlsx",
        "level": "class",
    },
    "unstable_dependency": {
        "display": "Unstable Dependency",
        "xlsx": "metrics_ud.xlsx",
        "level": "package",
    },
}

# Human phrasing injected into the agent prompt for the unit under test.
UNIT_KIND: dict[str, str] = {"package": "Java package", "class": "Java class"}

DEFAULT_PACKAGE = "(default package)"

# All definitions are filled; kept because run_batch.py's whole-matrix
# validation still asserts no placeholder slipped back in.
FILL_SENTINEL = "<<FILL"


def unit_for(smell: str, row: dict) -> str:
    """The unit identifier presented to the agent for one workbook row.

    Package-level smells -> the package FQN (`package` column).
    Class-level smells  -> the fully-qualified class name (`fqn` column).
    """
    if SMELL_META[smell]["level"] == "package":
        return (row.get("package") or "").strip()
    return (row.get("fqn") or "").strip()
