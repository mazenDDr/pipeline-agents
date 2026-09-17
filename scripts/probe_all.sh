#!/usr/bin/env bash
# T1 probe over every candidate, on the GPU machine. Probes whatever is on disk, then keeps
# looping while downloads are running, so each model is probed as soon as its file lands
# (finished rows are skipped). Run inside tmux via ./gpu run.
set -euo pipefail
RUN_ID=${RUN_ID:-t1-probe}
downloading() { tmux has-session -t claude-dl 2>/dev/null || tmux has-session -t claude-dl2 2>/dev/null; }

while true; do
  still_downloading=0; downloading && still_downloading=1
  python -m pipeline_agents.probe.run --run-id "$RUN_ID"
  [ "$still_downloading" = 0 ] && break  # a pass that started after the downloads ended saw every file
  sleep 60
done
python -m pipeline_agents.probe.report "outputs/runs/$RUN_ID"
