from smell_detection.state import SmellType, UnitKind

SMELL_DEFINITIONS: dict[SmellType, str] = {
    "IM": (
        "Insufficient Modularization occurs when a class has not been fully "
        "decomposed and could be split into smaller, more cohesive abstractions. "
        "Look for excessive size, many methods, many public methods, multiple "
        "responsibilities, low cohesion, and groups of methods that suggest "
        "separable concerns."
    ),
    "HM": (
        "Hub-like Modularization occurs when a class behaves as a dependency hub, "
        "having a high number of incoming and outgoing dependencies. Look for "
        "strong fan-in and fan-out evidence, many classes depending on the target, "
        "and the target depending on many other classes."
    ),
    "GC": (
        "God Component occurs when a package is excessively large or centralizes "
        "too many responsibilities. Look for many classes, high total size, broad "
        "responsibility concentration, and evidence that the package could be "
        "decomposed into more focused components."
    ),
    "UD": (
        "Unstable Dependency occurs when a package depends on packages that are "
        "less stable than itself. Stability should be reasoned from incoming and "
        "outgoing dependencies. A package with many incoming dependencies should "
        "not depend heavily on packages with fewer incoming dependencies and more "
        "outgoing dependencies."
    ),
}

# return the smell definition
def get_smell_definition(smell: SmellType) -> str:
    try:
        return SMELL_DEFINITIONS[smell]
    except KeyError as exc:
        raise ValueError(f"Unsupported smell type: {smell}") from exc

# check and return target type
def infer_target_type(smell: SmellType) -> UnitKind:
    if smell in {"IM", "HM"}:
        return "Java class"
    if smell in {"GC", "UD"}:
        return "Java package"

    raise ValueError(f"Cannot infer unit kind for smell: {smell}")