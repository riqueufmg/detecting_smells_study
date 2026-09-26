#!/usr/bin/env python3
"""Batch runner for the agentic architectural-smell detection experiment.

ONE invocation evaluates ONE smell with ONE model backbone. The smell candidates (packages or classes)
are extracted from:
    samples/metrics_<gc|hm|im|ud>.xlsx       (first sheet)

These are the same samples that were used in the previous experiments (RQ1).
For class-level smells (hublike_modularization, insufficient_modularization)
the workbook carries an explicit `fqn` column used verbatim as the unit;
ground truth is the `designite` column.

One mini-swe-agent trajectory JSON is written per cell to

    results/agentic/<smell>/<model_slug>/<project>/<unit_id>__repN.traj.json


Usage:
    python run_batch.py --smell hublike_modularization --model deepseek/deepseek-chat
    python run_batch.py --smell god_component --model qwen --project commons-lang --limit 5 --dry-run

`--model` is a full OpenRouter slug if it contains '/', otherwise an alias
resolved via the optional agentic/models.json ({"alias": "vendor/slug"}).
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
STUDY_ROOT = HERE.parent
SUBMODULE_SRC = STUDY_ROOT / "mini-swe-agent" / "src"
CONFIG_PATH = HERE / "smell_detect.yaml"
RESULTS_ROOT = STUDY_ROOT / "results" / "agentic"

SAMPLES_DIR = STUDY_ROOT / "samples"
TEMPLATES_DIR = STUDY_ROOT / "raw_code" / "data" / "prompts" / "templates"
REPOS_ROOT = HERE / "repos"
PINS_FILE = HERE / "repo_head_hashes.txt"
MODELS_FILE = HERE / "models.json"

# Use the pinned submodule without requiring `pip install`.
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SUBMODULE_SRC))

from minisweagent.config import get_config_from_spec  # noqa: E402
from minisweagent.environments import get_environment  # noqa: E402
from minisweagent.models import get_model  # noqa: E402
from minisweagent.utils.serialize import recursive_merge  # noqa: E402

from smell_agent import SmellAgent, SmellAgentConfig  # noqa: E402
from smell_definitions import (  # noqa: E402
    FILL_SENTINEL,
    SMELL_DEFINITIONS,
    SMELL_META,
    UNIT_KIND,
    unit_for,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("run_batch")

_TERMINAL_STATUSES = {"Submitted", "MalformedVerdict", "LimitsExceeded"}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _slug(text: str) -> str:
    """Filesystem-safe slug. Keeps dots/hyphens (FQNs), maps spaces,
    parentheses, slashes (e.g. an OpenRouter 'vendor/model' slug) to '_'."""
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in text)


def _unit_id(unit: str) -> str:
    """Canonical filesystem-safe id for a unit FQN (every non-alphanumeric
    character becomes '_'). Stable and identical to the zero short experiments RQ
    prompt-stem convention (e.g. `org.apache.commons.lang3.concurrent.locks`
    -> `org_apache_commons_lang3_concurrent_locks`)."""
    return "".join(c if c.isalnum() else "_" for c in unit)


def _git_sha(repo: Path) -> str:
    """git HEAD of `repo`, or '' if it cannot be determined."""
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def already_done(out_path: Path) -> bool:
    """Complete iff the trajectory exists and ended in a terminal status, so
    the batch is safely resumable."""
    if not out_path.exists():
        return False
    try:
        data = json.loads(out_path.read_text())
        return data.get("info", {}).get("exit_status", "") in _TERMINAL_STATUSES
    except Exception:
        return False


def cell_output_path(cell: dict, rep: int) -> Path:
    return (
        RESULTS_ROOT
        / _slug(cell["smell"])
        / _slug(cell["model_slug"])
        / _slug(cell["project"])
        / f"{_slug(cell['unit_id'])}__rep{rep}.traj.json"
    )


def error_marker_path(out_path: Path) -> Path:
    """The `.error.txt` marker file written next to the trajectory when a
    run crashes. Same directory and stem as the trajectory, just a
    different extension."""
    return out_path.parent / f"{out_path.stem}.error.txt"


def cell_rep_status(cell: dict, rep: int) -> str:
    """Classify one (cell, rep) by what is on disk:
        'done'         — terminal trajectory exists; nothing to do.
        'non_terminal' — trajectory exists but did not end terminally,
                         or is unparseable JSON.
        'errored'      — a previous attempt wrote a .error.txt marker
                         file and no trajectory followed.
        'missing'      — no trajectory and no error marker file.
    """
    out_path = cell_output_path(cell, rep)
    error_path = error_marker_path(out_path)
    if out_path.exists():
        try:
            data = json.loads(out_path.read_text())
            if data.get("info", {}).get("exit_status", "") in _TERMINAL_STATUSES:
                return "done"
        except Exception:
            pass
        return "non_terminal"
    return "errored" if error_path.exists() else "missing"


def partition_work(
    cells: list[dict], repetitions: int
) -> tuple[list[tuple[dict, int]], dict[str, int]]:
    """Walk the full (cell, rep) matrix and classify each pair on disk.
    Returns `(pending, counts)` where `pending` lists every pair that is
    not 'done' so the caller can run only the work that is left."""
    pending: list[tuple[dict, int]] = []
    counts = {"done": 0, "missing": 0, "errored": 0, "non_terminal": 0}
    for cell in cells:
        for rep in range(repetitions):
            status = cell_rep_status(cell, rep)
            counts[status] += 1
            if status != "done":
                pending.append((cell, rep))
    return pending, counts


# --------------------------------------------------------------------------- #
# resolution: model alias, repo pins, definition cross-check
# --------------------------------------------------------------------------- #
def resolve_model(arg: str) -> str:
    """A '/'-containing arg is taken as the OpenRouter slug verbatim;
    otherwise it is an alias looked up in agentic/models.json."""
    if "/" in arg:
        return arg
    if not MODELS_FILE.exists():
        sys.exit(
            f"--model '{arg}' is not a slug (no '/') and {MODELS_FILE.name} is "
            f"absent. Pass a full OpenRouter slug, or create agentic/models.json "
            f'like {{"{arg}": "vendor/model-slug"}}.'
        )
    aliases = json.loads(MODELS_FILE.read_text())
    if arg not in aliases:
        sys.exit(
            f"--model alias '{arg}' not in {MODELS_FILE.name}. "
            f"Known aliases: {sorted(aliases)}"
        )
    return aliases[arg]


def load_repo_pins() -> dict[str, tuple[str, str]]:
    """repo_head_hashes.txt ('owner/repo:sha') -> {repo_basename: (owner/repo, sha)}.
    The workbook `project` column equals the repo basename for every sampled repo."""
    if not PINS_FILE.exists():
        sys.exit(f"missing {PINS_FILE}")
    pins: dict[str, tuple[str, str]] = {}
    for raw in PINS_FILE.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        slug, _, sha = line.partition(":")
        if "/" not in slug or not sha:
            sys.exit(f"malformed pin line: {raw!r}")
        pins[slug.split("/")[-1]] = (slug, sha)
    return pins


def template_definition(smell: str) -> str | None:
    """The single line after `Definition:` in detection_<smell>.tpl, or None
    if the template (or that section) is unavailable."""
    tpl = TEMPLATES_DIR / f"detection_{smell}.tpl"
    if not tpl.exists():
        return None
    lines = tpl.read_text().splitlines()
    for i, ln in enumerate(lines):
        if ln.strip() == "Definition:":
            for nxt in lines[i + 1:]:
                if nxt.strip():
                    return nxt.strip()
            return None
    return None


def assert_definitions_match_templates() -> str:

    status = []
    mismatches = []
    for smell in SMELL_META:
        defn = SMELL_DEFINITIONS[smell]
        if FILL_SENTINEL in defn:
            mismatches.append(f"{smell}: definition still has a FILL placeholder")
            continue
        tpl_defn = template_definition(smell)
        if tpl_defn is None:
            log.warning("definition cross-check skipped (no template): %s", smell)
            status.append(f"{smell}:no-template")
            continue
        if tpl_defn != defn.strip():
            mismatches.append(
                f"{smell}: smell_definitions.py != template\n"
                f"    py : {defn.strip()!r}\n"
                f"    tpl: {tpl_defn!r}"
            )
        else:
            status.append(f"{smell}:ok")
    if mismatches:
        log.error("definition cross-check FAILED:")
        for m in mismatches:
            log.error("  - %s", m)
        sys.exit(1)
    log.info("definition cross-check: %s", " ".join(status))
    return ",".join(status)


# --------------------------------------------------------------------------- #
# matrix construction + validation (fails fast, before any container starts)
# --------------------------------------------------------------------------- #
def build_cells(
    smell: str, model_slug: str, projects: set[str] | None, limit: int | None
) -> list[dict]:
    xlsx_path = SAMPLES_DIR / SMELL_META[smell]["xlsx"]
    if not xlsx_path.exists():
        sys.exit(f"samples workbook not found: {xlsx_path}")
    pins = load_repo_pins()
    level = SMELL_META[smell]["level"]

    # First sheet, NaN -> None so the `(value or "")` idioms behave.
    df = pd.read_excel(xlsx_path, sheet_name=0)
    records = df.where(pd.notnull(df), None).to_dict(orient="records")

    cells: list[dict] = []
    errors: list[str] = []
    seen: dict[str, int] = {}
    for i, row in enumerate(records, 1):
        project = (row.get("project") or "").strip()
        if projects and project not in projects:
            continue
        unit = unit_for(smell, row)
        if not project or not unit:
            errors.append(f"{xlsx_path.name} row {i}: empty project/unit")
            continue
        unit_id = _unit_id(unit)
        if project not in pins:
            errors.append(
                f"{xlsx_path.name} row {i}: project '{project}' has no pin in "
                f"{PINS_FILE.name}"
            )
            continue
        owner_repo, sha = pins[project]
        cell_id = "__".join((smell, model_slug, project, unit_id))
        if cell_id in seen:
            errors.append(
                f"{xlsx_path.name} row {i}: duplicate cell_id '{cell_id}' "
                f"(also row {seen[cell_id]})"
            )
            continue
        seen[cell_id] = i
        try:
            designite = int(row.get("designite"))
        except (TypeError, ValueError):
            designite = None
        cells.append({
            "cell_id": cell_id,
            "smell": smell,
            "project": project,
            "unit": unit,
            "unit_kind": UNIT_KIND[level],
            "unit_id": unit_id,
            "model_slug": model_slug,
            "owner_repo": owner_repo,
            "repo_host_path": str(REPOS_ROOT / Path(owner_repo)),
            "commit_sha": sha,
            "designite": designite,
        })

    if errors:
        log.error("matrix rejected (%d problem(s)):", len(errors))
        for e in errors:
            log.error("  - %s", e)
        sys.exit(1)
    if limit is not None:
        cells = cells[:limit]
    if not cells:
        sys.exit("no cells selected (check --smell / --project / --limit)")
    return cells


def verify_repos(cells: list[dict]) -> dict[str, str]:
    """Each unique repo must exist and have HEAD == its pinned commit."""
    observed: dict[str, str] = {}
    errors: list[str] = []
    for cell in cells:
        repo = cell["repo_host_path"]
        if repo in observed:
            continue
        p = Path(repo)
        want = cell["commit_sha"]
        if not p.is_dir():
            errors.append(f"{repo} does not exist — run clone_repos.sh first")
            continue
        head = _git_sha(p)
        if not head:
            errors.append(f"{repo} is not a git repo / has no HEAD")
            continue
        if head != want:
            errors.append(
                f"{repo} HEAD is {head[:12]} but pin is {want[:12]} — "
                f"check out the pinned commit first"
            )
            continue
        observed[repo] = head
    if errors:
        log.error("repo verification FAILED (%d):", len(errors))
        for e in errors:
            log.error("  - %s", e)
        sys.exit(1)
    log.info("verified %d repo checkout(s)", len(observed))
    return observed


# --------------------------------------------------------------------------- #
# config assembly + run
# --------------------------------------------------------------------------- #
def base_config() -> dict:
    """builtin mini.yaml (model/observation/format plumbing) <- smell_detect.yaml."""
    return recursive_merge(get_config_from_spec("mini.yaml"), get_config_from_spec(CONFIG_PATH))


def cell_config(base: dict, cell: dict, out_path: Path) -> dict:
    """Per-cell overrides: OpenRouter slug, output path, read-only repo mount."""
    repo = Path(cell["repo_host_path"]).resolve()
    env = dict(base.get("environment", {}))
    env["run_args"] = list(env.get("run_args", [])) + ["-v", f"{repo}:/repo:ro"]
    return recursive_merge(
        base,
        {
            "model": {"model_name": cell["model_slug"]},
            "environment": env,
            "agent": {"output_path": str(out_path)},
        },
    )


def run_cell(base: dict, cell: dict, rep: int, observed_sha: str) -> str:
    out_path = cell_output_path(cell, rep)
    if already_done(out_path):
        log.info("skip (done): %s", out_path)
        return "skipped"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    config = cell_config(base, cell, out_path)
    definition = SMELL_DEFINITIONS[cell["smell"]]

    env = None
    try:
        model = get_model(config=config.get("model", {}))
        env = get_environment(config.get("environment", {}), default_type="docker")
        agent_kwargs = {
            k: v for k, v in config.get("agent", {}).items()
            if k in SmellAgentConfig.model_fields
        }
        agent = SmellAgent(model, env, **agent_kwargs)
        result = agent.run(
            task=f"Detect '{cell['smell']}' on {cell['unit_kind']} {cell['unit']}",
            # template vars used by smell_detect.yaml:
            smell_definition=definition,
            unit=cell["unit"],
            unit_kind=cell["unit_kind"],
            repo_path="/repo",
            # provenance embedded into the trajectory by SmellAgent.serialize:
            cell_id=cell["cell_id"],
            project=cell["project"],
            smell=cell["smell"],
            unit_id=cell["unit_id"],
            model_slug=cell["model_slug"],
            repo_host_path=cell["repo_host_path"],
            expected_commit_sha=cell["commit_sha"],
            observed_commit_sha=observed_sha,
            repetition=rep,
            designite=cell["designite"],
        )
        status = result.get("exit_status", "unknown")
        # On a successful retry, the .error.txt marker file from the prior
        # crashed attempt is no longer meaningful — drop it.
        if status in _TERMINAL_STATUSES:
            marker = error_marker_path(out_path)
            if marker.exists():
                marker.unlink()
        log.info("done [%s]: %s", status, out_path)
        return status
    except Exception as e:
        log.error("FAILED %s: %s", out_path, e)
        error_marker_path(out_path).write_text(
            f"{e}\n\n{traceback.format_exc()}"
        )
        return "error"
    finally:
        if env is not None and hasattr(env, "cleanup"):
            env.cleanup()


def write_run_manifest(
    smell: str, model_slug: str, base: dict, cells: list[dict],
    observed: dict[str, str], k: int, defn_check: str,
) -> None:
    """Provenance for the run. No content hashing: the agentic/ source is
    pinned by the study-repo git commit; the scaffold by the submodule SHA.
    `base` is the resolved base config (post any CLI overrides such as
    `--provider`) so the manifest reflects exactly what is sent to the LM."""
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "smell": smell,
        "model_slug": model_slug,
        "samples_xlsx": str(
            (SAMPLES_DIR / SMELL_META[smell]["xlsx"]).relative_to(STUDY_ROOT)
        ),
        "definition_cross_check": defn_check,
        "mini_swe_agent_submodule_sha": _git_sha(STUDY_ROOT / "mini-swe-agent"),
        "study_repo_sha": _git_sha(STUDY_ROOT),
        "study_repo_dirty": bool(
            subprocess.run(
                ["git", "-C", str(STUDY_ROOT), "status", "--porcelain"],
                capture_output=True, text=True,
            ).stdout.strip()
        ),
        "docker_image": base.get("environment", {}).get("image"),
        "step_limit": base.get("agent", {}).get("step_limit"),
        "cost_limit": base.get("agent", {}).get("cost_limit"),
        "max_verdict_retries": base.get("agent", {}).get("max_verdict_retries"),
        "model_kwargs": base.get("model", {}).get("model_kwargs"),
        "repetitions": k,
        "n_cells": len(cells),
        "projects": sorted({c["project"] for c in cells}),
        "repo_checkouts": {r: observed[r] for r in sorted(observed)},
    }
    out_dir = RESULTS_ROOT / _slug(smell) / _slug(model_slug)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "_run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    log.info("wrote reproducibility manifest: %s", path)


def print_matrix(cells: list[dict]) -> None:
    log.info("DRY RUN — %d cells:", len(cells))
    for c in cells:
        print(
            f"  {c['project']:<18} {c['unit_kind']:<12} designite={c['designite']}  "
            f"{c['unit']}  @ {c['commit_sha'][:12]}"
        )


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--smell", required=True, choices=sorted(SMELL_META))
    p.add_argument("--model", required=True, help="OpenRouter slug, or alias from models.json")
    p.add_argument("--project", action="append", help="restrict to project(s); repeatable")
    p.add_argument("--repetitions", type=int, default=1)
    p.add_argument("--limit", type=int, help="cap #cells (pilot/debug)")
    p.add_argument("--dry-run", action="store_true", help="print the matrix and exit")
    p.add_argument(
        "--provider", action="append", metavar="NAME",
        help="pin OpenRouter to this provider (repeatable for a priority "
             "list, e.g. '--provider Fireworks --provider Together'). "
             "See https://openrouter.ai/docs/features/provider-routing",
    )
    p.add_argument(
        "--allow-provider-fallback", action="store_true",
        help="when --provider is set, allow OpenRouter to fall back to "
             "providers outside the listed order if none can serve the "
             "request (default: strict — fail rather than fall back)",
    )
    args = p.parse_args()

    defn_check = assert_definitions_match_templates()
    model_slug = resolve_model(args.model)
    cells = build_cells(
        args.smell, model_slug,
        set(args.project) if args.project else None,
        args.limit,
    )

    base_cfg = base_config()
    if args.provider:
        provider_block = {
            "order": list(args.provider),
            "allow_fallbacks": bool(args.allow_provider_fallback),
        }
        base_cfg.setdefault("model", {}).setdefault("model_kwargs", {})["provider"] = provider_block
        log.info(
            "openrouter provider pin: order=%s allow_fallbacks=%s",
            provider_block["order"], provider_block["allow_fallbacks"],
        )

    pending, counts = partition_work(cells, args.repetitions)
    total = sum(counts.values())
    log.info(
        "resume status [%s / %s]: done=%d/%d, pending=%d "
        "(missing=%d, errored=%d, non_terminal=%d)",
        args.smell, model_slug, counts["done"], total, len(pending),
        counts["missing"], counts["errored"], counts["non_terminal"],
    )

    if args.dry_run:
        print_matrix(cells)
        return

    if not pending:
        log.info("nothing to do — all (cell, rep) pairs are already complete")
        return

    observed = verify_repos(cells)
    write_run_manifest(args.smell, model_slug, base_cfg, cells, observed, args.repetitions, defn_check)

    summary: dict[str, int] = {}
    for cell, rep in pending:
        status = run_cell(base_cfg, cell, rep, observed[cell["repo_host_path"]])
        summary[status] = summary.get(status, 0) + 1
    log.info("BATCH SUMMARY [%s / %s]: %s", args.smell, model_slug, summary)


if __name__ == "__main__":
    main()
