# Architecture Smell Detection using Agents

Agentic evaluation of LLMs on architectural code-smell detection. Each run is one
[mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) invocation that
decides whether a specified smell affects a specified Java package or class, by
exploring the source through a read-only shell inside a sandboxed Docker
container.

Smells under evaluation:
`god_component`, `hublike_modularization`, `insufficient_modularization`,
`unstable_dependency`.

## Prerequisites

- Python 3.10+
- Docker (the agent runs in a `--network none` container)
- An [OpenRouter](https://openrouter.ai) API key (models are routed via OpenRouter)
- Git (to fetch projects under test at their pinned commits)

## Setup

```bash
# 1. Clone, including the mini-swe-agent submodule
git clone --recursive <repo-url>
cd Architecture-Smell-Detection-using-Agents

# 2. Create a venv and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -e mini-swe-agent
pip install pandas openpyxl

# 3. Clone the 12 Java projects under test at their pinned commits.
#    Output: agentic/repos/<owner>/<repo>
./agentic/clone_repos.sh

# 4. Export your OpenRouter key
export OPENROUTER_API_KEY=sk-or-...
```

The pinned commits are listed in `agentic/repo_head_hashes.txt`. The script `clone_repos.sh` clones the repositories and checks out each to the specified commit.

## Running an experiment

One invocation evaluates **one smell** with **one model** over the samples. The set of samples are located in Excel sheets: `samples/metrics_<gc|hm|im|ud>.xlsx`.

```bash
# Preview the run
python agentic/run_batch.py --smell hublike_modularization \
    --model deepseek/deepseek-chat --limit 5 --dry-run

# Full run
python agentic/run_batch.py --smell hublike_modularization \
    --model deepseek/deepseek-chat --repetitions 5
```

Flags:

| flag | meaning |
|---|---|
| `--smell` | one of the four smell keys (required) |
| `--model` | full OpenRouter slug (e.g. `deepseek/deepseek-chat`), or an alias from optional `agentic/models.json` |
| `--repetitions K` | number of times the experiment must be repeated (default `1`) |
| `--project P` | restrict to a project from the list of 12 projects |
| `--limit N` | first `N` instances. Useful for debugging or smoke testing. |
| `--dry-run` | print which packages/classes would be evaluated and exits. It does not spawn Docker containers. |

Trajectories are written to:

```
results/agentic/<smell>/<model_slug>/<project>/<unit_id>__repN.traj.json
```

alongside a `_run_manifest.json` capturing the submodule SHA, study-repo SHA,
docker image, step/cost limits, and the verified per-repo commits.

## Evaluation

```bash
python agentic/score.py --smell hublike_modularization \
    --model deepseek/deepseek-chat --repetitions 5
```

Emits two CSVs in the same `(smell, model)` directory:

- **`predictions_long.csv`** — one row per `(cell, repetition)`: `exit_status`,
  `verdict_valid`, `detection_binary`, `verdict_error`, `designite`. Use for
  repeated-measures / random-effects analyses.
- **`predictions_agg.csv`** — one row per cell: `designite` (ground truth),
  `detection_mean`, `detection_majority` (`null` on a tie), and exclusion
  counts (`n_missing`, `n_malformed`, `n_limits`, `n_error`).

Metrics (accuracy, precision/recall, F1, …) are intentionally **not** computed
inside this repo: join `detection_majority` (or threshold `detection_mean`)
against `designite` with the metric library of your choice.

## Repository layout

```
agentic/                       runner, scorer, agent subclass, prompt config
  run_batch.py                   build matrix from samples/*.xlsx, run agent
  score.py                       trajectories -> predictions_long/agg.csv
  smell_agent.py                 DefaultAgent subclass with verdict-schema gate
  smell_definitions.py           canonical smell definitions + per-smell metadata
  smell_detect.yaml              system/instance templates, env, model settings
  clone_repos.sh                 fetch projects under test at pinned commits
  repo_head_hashes.txt           owner/repo:sha pins (12 projects)
samples/                       pre-registered candidate workbooks (one per smell)
raw_code/data/prompts/templates/  static detection prompts; their `Definition:`
                                  is cross-checked against smell_definitions.py
                                  at launch and the run aborts on drift
mini-swe-agent/                agent scaffold (git submodule)
agentic/repos/                 cloned projects under test (created by clone_repos.sh)
results/agentic/               trajectories + predictions CSVs + manifest
```
