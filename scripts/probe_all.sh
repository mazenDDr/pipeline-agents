#!/usr/bin/env bash
# T1 probe over every candidate, on the GPU machine: probe what is on disk now, wait for the
# downloads, then probe again (finished rows are skipped). Run inside tmux via ./gpu run.
set -euo pipefail
RUN=outputs/runs/${RUN_ID:-t1-probe}
wait_for() { while tmux has-session -t "$1" 2>/dev/null; do sleep 60; done; }

wait_for claude-probe  # the 4B smoke run uses the same code and settings; reuse its rows
mkdir -p "$RUN"
for f in results calls servers; do
  [ -f "outputs/runs/smoke-4b/$f.jsonl" ] && [ ! -f "$RUN/$f.jsonl" ] && cp "outputs/runs/smoke-4b/$f.jsonl" "$RUN/"
done

python -m pipeline_agents.probe.run --run-id "$(basename "$RUN")"
wait_for claude-dl
wait_for claude-dl2
python -m pipeline_agents.probe.run --run-id "$(basename "$RUN")"
python -m pipeline_agents.probe.report "$RUN"
