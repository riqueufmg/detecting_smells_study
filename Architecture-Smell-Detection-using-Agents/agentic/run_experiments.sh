#!/usr/bin/env bash
# Run all four smells for ONE OpenRouter model in the background, sequentially.
#
# Each smell is one `python agentic/run_batch.py` invocation; the four runs
# happen one after the other to keep Docker and OpenRouter load bounded.
# Resumability is handled by run_batch.py itself — re-running this script
# skips any cells/reps that already produced a terminal trajectory.
#
# Usage:
#     ./agentic/run_experiments.sh <openrouter-slug> [repetitions]
#
# Examples:
#     ./agentic/run_experiments.sh deepseek/deepseek-chat
#     ./agentic/run_experiments.sh openai/gpt-4o 3
#
# Environment overrides:
#     PYTHON   interpreter to use (default: <repo-root>/.venv/bin/python)

set -euo pipefail

SMELLS=(god_component hublike_modularization insufficient_modularization unstable_dependency)

# Filesystem-safe slug (mirrors _slug in run_batch.py).
slug() { printf '%s' "$1" | tr -c 'A-Za-z0-9._-' '_'; }

# ---------------------------------------------------------------------------
# Worker mode: re-invoked under nohup. Foreground inside its own process; the
# foreground caller has already detached us.
# ---------------------------------------------------------------------------
if [[ "${1:-}" == "__worker" ]]; then
    shift
    MODEL="$1" REPS="$2" LOGDIR="$3" REPO_ROOT="$4" PY="$5"
    exec >> "$LOGDIR/_driver.log" 2>&1
    cat <<HEADER
============================================================
agentic experiment run
============================================================
  model:                $MODEL
  repetitions / smell:  $REPS
  smells:               ${SMELLS[*]}
  pid:                  $$
  python:               $PY
  driver log:           $LOGDIR/_driver.log
  per-smell logs:       $LOGDIR/<smell>.log
  started:              $(date -Iseconds)
============================================================
HEADER
    final_rc=0
    for s in "${SMELLS[@]}"; do
        echo "[$(date -Iseconds)] starting $s"
        if "$PY" "$REPO_ROOT/agentic/run_batch.py" \
                --smell "$s" --model "$MODEL" --repetitions "$REPS" \
                > "$LOGDIR/$s.log" 2>&1; then
            echo "[$(date -Iseconds)] finished $s ok"
        else
            rc=$?
            echo "[$(date -Iseconds)] finished $s FAILED rc=$rc  (see $LOGDIR/$s.log)"
            final_rc=$rc
        fi
    done
    echo "[$(date -Iseconds)] driver done (final_rc=$final_rc)"
    exit "$final_rc"
fi

# ---------------------------------------------------------------------------
# Foreground: validate inputs, fork the worker, print how to follow/stop it.
# ---------------------------------------------------------------------------
MODEL="${1:?usage: $0 <openrouter-slug> [repetitions]}"
REPS="${2:-5}"

SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
HERE="$(dirname "$SELF")"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
PY="${PYTHON:-$REPO_ROOT/.venv/bin/python}"

[[ -x "$PY" ]] || { echo "error: $PY is not executable (set PYTHON=...)" >&2; exit 1; }
[[ -n "${OPENROUTER_API_KEY:-}" ]] || { echo "error: OPENROUTER_API_KEY is not set" >&2; exit 1; }

LOGDIR="$REPO_ROOT/logs/$(slug "$MODEL")"
mkdir -p "$LOGDIR"

nohup "$SELF" __worker "$MODEL" "$REPS" "$LOGDIR" "$REPO_ROOT" "$PY" >/dev/null 2>&1 &
PID=$!
disown "$PID" 2>/dev/null || true
echo "$PID" > "$LOGDIR/_driver.pid"

cat <<EOF
started background run
  model:                $MODEL
  repetitions / smell:  $REPS
  pid:                  $PID
  driver log:           $LOGDIR/_driver.log
  per-smell logs:       $LOGDIR/<smell>.log

follow progress:  tail -f "$LOGDIR/_driver.log"
stop:             kill \$(cat "$LOGDIR/_driver.pid")
                  # if the python child outlives the worker, also:
                  # pkill -f "run_batch.py --model $MODEL"
EOF
