#!/usr/bin/env python3
"""Score agentic trajectories: produce predictions tables ready for metrics.

Given (smell, model, K reps, trajectory tree), walks every cell in the
pre-registered matrix (same path run_batch.py uses to construct it) and for
each (cell, repetition) extracts the binary detection from the trajectory:

  - exit_status=Submitted + verdict_valid=True   -> detection_binary in {0,1}
  - exit_status=MalformedVerdict (out of retries) -> excluded (off-schema)
  - exit_status=LimitsExceeded                    -> excluded (no submission)
  - trajectory file missing / unreadable         -> excluded

Two CSVs are written into the (smell, model) results directory:

  predictions_long.csv  - one row per (cell, repetition); raw view, for
                         random-effects / mixed-effects analyses.
  predictions_agg.csv   - one row per cell with majority vote and mean
                         across valid reps, for the cell-level metrics
                         (acc, precision/recall/F1 against `designite`).

This file is a DETERMINISTIC PROJECTION. It does NOT compute metrics; the
predictions are emitted as binary columns and the ground truth as the
`designite` column, so any downstream tool/library can compute whatever
metric is wanted.

Aggregation policy (documented and re-derivable from the long CSV):
  - valid reps = those with verdict_valid=True and detection in {0,1}
  - detection_mean = mean of valid reps (null if zero valid reps)
  - detection_majority = 1 if mean>0.5, 0 if mean<0.5, null on tie / no valid

Usage:
    python score.py --smell hublike_modularization --model deepseek/deepseek-chat
    python score.py --smell god_component --model qwen --repetitions 5 --project commons-lang
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from smell_definitions import SMELL_META  # noqa: E402
from run_batch import (  # noqa: E402
    RESULTS_ROOT,
    SAMPLES_DIR,
    _slug,
    build_cells,
    cell_output_path,
    resolve_model,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("score")


def load_trajectory(path: Path) -> dict | None:
    """Return None if the trajectory file is absent. Otherwise a dict of
    the four fields the scorer cares about. Survives partially-written /
    corrupt JSON by reporting exit_status='read_error'."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except Exception as e:
        return {"exit_status": "read_error", "verdict_valid": None,
                "detection_binary": None, "verdict_error": str(e)}
    info = data.get("info") or {}
    msgs = data.get("messages") or []
    extra = (msgs[-1].get("extra") if msgs else {}) or {}
    exit_status = info.get("exit_status") or extra.get("exit_status") or "unknown"
    verdict_valid = extra.get("verdict_valid")
    verdict = extra.get("verdict") if isinstance(extra.get("verdict"), dict) else None
    detection = verdict.get("detection") if verdict else None
    return {
        "exit_status": exit_status,
        "verdict_valid": bool(verdict_valid) if verdict_valid is not None else None,
        "detection_binary": int(detection) if isinstance(detection, bool) else None,
        "verdict_error": extra.get("verdict_error"),
    }


def _aggregate(per_rep: list[dict], repetitions: int) -> dict:
    valid = [r["detection_binary"] for r in per_rep
             if r["verdict_valid"] is True and r["detection_binary"] in (0, 1)]
    n_valid = len(valid)
    mean = (sum(valid) / n_valid) if n_valid else None
    majority = None
    if mean is not None:
        if mean > 0.5:
            majority = 1
        elif mean < 0.5:
            majority = 0
        # exact tie -> None on purpose; downstream chooses a tie-break
    return {
        "n_reps_total": repetitions,
        "n_reps_valid": n_valid,
        "n_missing":    sum(1 for r in per_rep if r["exit_status"] is None),
        "n_malformed":  sum(1 for r in per_rep if r["exit_status"] == "MalformedVerdict"),
        "n_limits":     sum(1 for r in per_rep if r["exit_status"] == "LimitsExceeded"),
        "n_error":      sum(1 for r in per_rep
                            if r["exit_status"] in ("error", "read_error", "unknown")),
        "detection_mean": mean,
        "detection_majority": majority,
    }


def score(smell: str, model_arg: str, repetitions: int,
          projects: set[str] | None, out_dir: Path | None,
          ground_truth: Path | None) -> tuple[Path, Path]:
    model_slug = resolve_model(model_arg)
    cells = build_cells(smell, model_slug, projects, limit=None)

    # Sanity-check that --ground-truth (if given) matches the workbook the
    # matrix was built from. The `designite` column written below comes from
    # build_cells (i.e. the registered workbook); a mismatched override does
    # not silently swap it.
    expected_gt = SAMPLES_DIR / SMELL_META[smell]["xlsx"]
    if ground_truth and ground_truth.resolve() != expected_gt.resolve():
        log.warning("--ground-truth %s != registered workbook %s; the "
                    "registered workbook is used for the `designite` column.",
                    ground_truth, expected_gt)

    long_rows: list[dict] = []
    agg_rows: list[dict] = []
    tally = {"valid": 0, "malformed": 0, "limits": 0, "error": 0, "missing": 0, "unknown": 0}

    for cell in cells:
        per_rep: list[dict] = []
        for r in range(repetitions):
            traj = load_trajectory(cell_output_path(cell, r))
            row = {
                "cell_id": cell["cell_id"],
                "smell": cell["smell"],
                "project": cell["project"],
                "unit": cell["unit"],
                "unit_id": cell["unit_id"],
                "model_slug": cell["model_slug"],
                "repetition": r,
                "designite": cell["designite"],
                "exit_status": None if traj is None else traj["exit_status"],
                "verdict_valid": None if traj is None else traj["verdict_valid"],
                "detection_binary": None if traj is None else traj["detection_binary"],
                "verdict_error": None if traj is None else traj["verdict_error"],
            }
            long_rows.append(row)
            per_rep.append(row)

            if traj is None:
                tally["missing"] += 1
            elif traj["exit_status"] == "Submitted" and traj["verdict_valid"]:
                tally["valid"] += 1
            elif traj["exit_status"] == "MalformedVerdict":
                tally["malformed"] += 1
            elif traj["exit_status"] == "LimitsExceeded":
                tally["limits"] += 1
            elif traj["exit_status"] in ("read_error", "error"):
                tally["error"] += 1
            else:
                tally["unknown"] += 1

        agg = _aggregate(per_rep, repetitions)
        agg_rows.append({
            "cell_id": cell["cell_id"],
            "smell": cell["smell"],
            "project": cell["project"],
            "unit": cell["unit"],
            "unit_id": cell["unit_id"],
            "model_slug": cell["model_slug"],
            "designite": cell["designite"],
            **agg,
        })

    out_dir = out_dir or (RESULTS_ROOT / _slug(smell) / _slug(model_slug))
    out_dir.mkdir(parents=True, exist_ok=True)
    long_path = out_dir / "predictions_long.csv"
    agg_path = out_dir / "predictions_agg.csv"
    pd.DataFrame(long_rows).to_csv(long_path, index=False)
    pd.DataFrame(agg_rows).to_csv(agg_path, index=False)

    log.info("wrote %s (%d rows)", long_path, len(long_rows))
    log.info("wrote %s (%d cells)", agg_path, len(agg_rows))
    log.info("trajectory tally: %s", tally)
    return long_path, agg_path


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--smell", required=True, choices=sorted(SMELL_META))
    p.add_argument("--model", required=True, help="OpenRouter slug, or alias from models.json")
    p.add_argument("--repetitions", type=int, default=5)
    p.add_argument("--project", action="append", help="restrict to project(s); repeatable")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="default: results/agentic/<smell>/<model_slug>/")
    p.add_argument("--ground-truth", type=Path, default=None,
                   help="cross-check only; predictions still come from trajectories")
    args = p.parse_args()
    score(
        args.smell, args.model, args.repetitions,
        set(args.project) if args.project else None,
        args.out_dir, args.ground_truth,
    )


if __name__ == "__main__":
    main()
