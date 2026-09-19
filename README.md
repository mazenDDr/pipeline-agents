# pipeline-agents

An experiment in whether a team of specialized agents can build safer pandas and scikit-learn pipelines
than one model working alone.

The team plans, executes one step at a time, checks the files in a sandbox, revises failures, and can re-plan.
Every model call is logged, while typed run state and validation results are checkpointed after every graph
node. The answer
from the experiment is useful precisely because it is not the expected one: **the single-agent baseline won
the controlled test.**

<img src="docs/img/results.svg" width="100%" alt="Test success rates: the single-agent baseline scored 0.88, followed by the agent team with all memories at 0.71; the team without memory scored 0.50.">

On eight held-out benchmark tasks and three seeds, the baseline succeeded **0.88** of the time, compared with
**0.50** for the team without memory. The paired difference was +0.38 [0.04, 0.71]. Giving the team all three
memory stores raised its result to **0.71**, but the +0.21 difference from no memory was not distinguishable
from noise. Costs below are *shadow prices*—token counts multiplied by a dated public price table—not money
spent; all models ran locally.

| system | success [95% CI] | model calls | shadow $ | wall time |
|---|---:|---:|---:|---:|
| one model, whole pipeline | 0.88 [0.67, 1.00] | 1.8 | 0.0016 | 153 s |
| agent team, all memories | 0.71 [0.46, 0.92] | 20.9 | 0.0093 | 363 s |
| agent team, no memory | 0.50 [0.21, 0.79] | 23.0 | 0.0097 | 373 s |

The full tables, paired comparisons, per-task outcomes, and confounds are in
[the ablation report](docs/ablations.md). On three public datasets the benchmark had never seen, both systems
passed every task (6/6 runs), including a two-million-row file; the baseline still used 8–9x fewer model calls.
See [the field test](docs/field_test.md).

## How it works

<img src="docs/img/loop.svg" width="100%" alt="The exact LangGraph: init, plan, execute, validate, critic and advance; failures go through revise, accepted plans go through deliver, and benchmark runs then use a separate hidden checker.">

The diagram follows [`graph/build.py`](src/pipeline_agents/graph/build.py), including its actual node names and
conditional edges:

| node | what the code does | next |
|---|---|---|
| `init` | profile `data/`; retrieve semantic facts by file-schema fingerprint and similar past episodes | `plan` |
| `plan` | the Planner returns a typed `Plan` | `execute` |
| `execute` | retrieve procedural skills for this step; the Executor writes one Python file and the sandbox runs it | `validate`, retry infrastructure failure, or `revise` a crash |
| `validate` | deterministic checks inspect the artifacts, never the Executor's claims | `critic` |
| `critic` | the Critic returns a typed `Verdict`: accept, revise, or escalate | `advance`, `revise`, or `human` on low confidence |
| `advance` | mark the step accepted and move the cursor | next `execute`, or `deliver` after the last step |
| `revise` | the Reviser returns fix instructions or escalates the plan | retry `execute`, re-enter `plan`, or `human` when re-plans are exhausted |
| `deliver` | assemble accepted step files into `pipeline.py`; clean-rerun it and smoke-test `predict.py` | `succeeded`, or `revise` the last step |
| `human` | stop as `needs_human` with the reason | end |

The validation node checks readable tables, numeric columns stored as text, missing/sentinel values, row loss,
required IDs, target leakage, split overlap, temporal order, metric sanity, and JSON outputs when applicable.
The Linux sandbox is a systemd scope plus user/network/mount/PID namespaces and resource limits: input data is
read-only, `output/` is writable, and the repository, other runs, network, and hidden answers are absent.

Two cross-cutting pieces sit around the graph:

- `RunState` is one Pydantic object checkpointed to SQLite after every node. It holds the plan history, every
  attempt/verdict/revision, retrieved memory, spend ledger, status, and stop reason.
- The `Observer` wraps every model call. It chooses the budget tier and appends the rendered messages, prompt
  versions and hashes, raw content/reasoning, token counts, latency, model, tier, cache key, error, and shadow
  cost to `calls.jsonl`.

The hidden checker is deliberately **outside** the graph in [`bench/run.py`](src/pipeline_agents/bench/run.py).
Only after the run stops can the benchmark runner expose its output to hidden labels and answers. The
single-agent baseline bypasses LangGraph and per-step validation, but shares the same model harness, budget,
sandbox, delivery checks, repair allowance, and hidden checker. That is the comparison being measured.

[Walk through one real 37-call run](docs/tour.md), including its saved code, tool findings, rejected attempt,
re-plan, and hidden-check result.

## What was measured

The benchmark has 14 tasks from seven UCI dataset families. Families—not individual rows—are split between
development and test so related tasks cannot leak across the boundary. A run succeeds only if its output:

1. runs from a clean workspace;
2. passes every applicable trap check; and
3. beats a threshold set by a reference solution on the hidden holdout.

The test comparison is 8 tasks × 3 seeds × 6 configurations: the single-agent baseline, the team without
memory, each memory alone, and all memories together. Intervals use a nested bootstrap over tasks and seeds;
differences use paired runs. Prompts and loop settings were screened on development tasks only.

This evaluation also checks its checkers. The benchmark accepted and rejected 58/58 known-good and deliberately
broken solutions. A hand audit confirmed 15/15 sampled run failures, while the model Critic agreed only weakly
with careful labels (Cohen's κ 0.30 [0.04, 0.54]). Critic decisions are therefore diagnostics, not ground truth.

Read the evidence:

- [Benchmark design and traps](docs/benchmark.md)
- [Model and serving-runtime measurements](docs/model_choice.md)
- [Held-out ablations](docs/ablations.md)
- [Failure taxonomy for 94 failed runs](docs/failure_taxonomy.md)
- [Critic and checker trust audit](docs/trust.md)
- [Fresh-data field test](docs/field_test.md)

## Run it

Requirements: Python 3.11, Linux for the full sandbox, and OpenAI-compatible local model endpoints matching
`configs/models.yaml`. The measured setup used `llama-server` and an RTX 5060 Ti with 16 GB VRAM. CPU unit
tests use scripted model responses and do not download models or call an endpoint.

```bash
python3.11 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev,app]"
make check
```

The repository's measured setup uses the included `./gpu` helper and an SSH/Tailscale host named `gpu-box`.
Start the two configured model servers in a remote tmux session:

```bash
GPU_SESSION=serve ./gpu run bash scripts/serve_models.sh tiered
```

In another terminal, download the UCI sources once, build and validate the benchmark, then run one task:

```bash
./gpu exec scripts/fetch_datasets.py
./gpu exec scripts/validate_benchmark.py
./gpu exec scripts/run_task.py --task bike-2-weather --system multi
```

The full validation proves all 58 reference, honest, and deliberately broken solutions behave as expected;
it can take a few minutes. Use `--system baseline` for the one-model comparison. `scripts/run_grid.py` runs
resumable experiment grids, and `configs/grids/` contains the exact committed configurations. Raw datasets,
built workspaces, hidden answers, model weights, and full run workspaces are intentionally not committed.

To run on another Linux machine, change `server_binary` and `model_dir` in `configs/models.yaml`, serve the two
OpenAI-compatible endpoints on ports 8081 and 8082, and set `PIPELINE_MODEL_HOST` when they are not localhost.

To open the Streamlit interface after the model endpoints are healthy:

```bash
PIPELINE_MODEL_HOST=<gpu-box-address> make app
```

It accepts a CSV and goal, streams the plan and checks, and exposes finished-run and aggregate dashboards.

## Repository map

```text
configs/                 model, run, price, and experiment-grid settings
prompts/                 immutable, versioned prompts and their lock file
src/pipeline_agents/
  agents/                team roles and the single-agent baseline
  graph/                 LangGraph orchestration and delivery checks
  harness/               model client, prompt renderer, budget, observer
  sandbox/               isolated process runner and failure classification
  tools/                 dataset profiler and deterministic validators
  memory/                procedural, semantic, and episodic stores
  bench/                 benchmark builder, runner, checker, and field tests
  eval/                  bootstrap comparisons and trust audit
  app/                   Streamlit interface
benchmark/solutions/     references plus deliberately broken controls
outputs/                 committed summaries and generated reports
scripts/                 reproducible CLIs and document builders
tests/                   fast CPU tests with a scripted fake model
```

## Limits and next experiments

- Eight test tasks leave wide intervals; differences smaller than roughly 0.2 are unresolved.
- Seven team runs exceeded the 4 GB sandbox limit, unevenly across memory arms. They remain failures in the
  reported result.
- Two benchmark/tool defects were found after the runs and are disclosed in the report rather than silently
  recomputed away.
- The team often found a real problem but could not repair a file owned by an earlier step. A whole-pipeline
  repair path and routing memory-limit failures to the Reviser are hypotheses for a new development experiment.
- The selected models are local open weights. These numbers do not establish the same ranking for larger or
  hosted models.

The result is not that agents never help. It is narrower: in this system, on this measured task set, extra
roles and repair loops cost more and succeeded less often than letting one capable model rewrite the whole
pipeline. The stored traces make the reason inspectable instead of speculative.

## Rebuild the figures

The README figures are self-contained SVGs generated from code. The result chart reads the committed T11
summaries; its values are not copied into the drawing script.

```bash
python scripts/build_svgs.py
```
