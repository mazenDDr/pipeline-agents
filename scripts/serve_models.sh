#!/usr/bin/env bash
# Serve a layout from configs/models.yaml on the GPU machine: strong first (so --fit sizes it with the
# cheap model's room in mind), then cheap, then wait. Run inside tmux: GPU_SESSION=serve ./gpu run bash scripts/serve_models.sh tiered
set -euo pipefail
LAYOUT=${1:-tiered}
python - "$LAYOUT" > outputs/serve-commands.txt <<'PY'
import shlex, sys, yaml
cfg = yaml.safe_load(open("configs/models.yaml"))
seen = set()
for tier in ("strong", "cheap"):
    spec = cfg["layouts"][sys.argv[1]][tier]
    if spec["port"] in seen:
        continue
    seen.add(spec["port"])
    model = cfg["models"][spec["model"]]
    cmd = [cfg["server_binary"], "-m", f"{cfg['model_dir']}/{model['file']}", "--host", "0.0.0.0",
           "--port", str(spec["port"]), *cfg["common_args"], *spec["args"]]
    print(spec["port"], shlex.join(cmd))
PY
pids=()
while read -r port cmd; do
  echo "starting on :$port -> $cmd"
  eval "$cmd" > "outputs/serve-$port.log" 2>&1 &
  pids+=($!)
  until curl -sf "localhost:$port/health" > /dev/null; do
    kill -0 "${pids[-1]}" 2>/dev/null || { echo "server on :$port died; see outputs/serve-$port.log"; exit 1; }
    sleep 1
  done
  echo ":$port healthy"
done < outputs/serve-commands.txt
trap 'kill "${pids[@]}" 2>/dev/null' EXIT
wait
