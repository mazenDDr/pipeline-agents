"""Can a strong and a cheap model be served together on one 16 GB card? (T1)

For each layout, start both llama-servers, then time the same 6k-token prompt on each (prompt cache
off, 3 repeats), and record VRAM, both servers' resident memory and the RAM left for the sandbox.

    python scripts/colocate_check.py            (on the GPU machine; writes outputs/runs/t1-colocate/)
"""

import json
import statistics
import subprocess
import time
from pathlib import Path

from openai import OpenAI

from pipeline_agents.probe.run import SPEED_PROMPT, Server, chat, idle_vram_mib

BIN = "/home/mazen/llama.cpp/build-cuda/bin/llama-server"
MODELS = Path("models/gguf")
COMMON = ["-c", "16384", "-np", "1", "-fa", "on", "--jinja", "--reasoning-format", "deepseek"]
STRONG = ("gemma-4-26b-a4b", "gemma-4-26B-A4B-it-UD-IQ4_XS.gguf")
CPU = ["-ngl", "0", "-t", "8"]

LAYOUTS = {
    "strong-alone": [(STRONG, ["--fit", "on"])],
    "both-gpu": [
        (STRONG, ["--fit", "on", "--fit-target", "5120"]),  # leave room for the cheap model
        (("gemma-4-e4b", "gemma-4-E4B-it-Q4_K_M.gguf"), ["--fit", "on"]),
    ],
    "cheap-e4b-cpu": [(STRONG, ["--fit", "on"]), (("gemma-4-e4b", "gemma-4-E4B-it-Q4_K_M.gguf"), CPU)],
    "cheap-qwen4b-cpu": [(STRONG, ["--fit", "on"]), (("qwen3.5-4b", "Qwen3.5-4B-Q4_K_M.gguf"), CPU)],
}
NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}


def rss_mib(pid: int) -> int:
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) // 1024
    return 0


def available_ram_mib() -> int:
    out = subprocess.run(["free", "-m"], capture_output=True, text=True).stdout.splitlines()[1].split()
    return int(out[6])


def main() -> None:
    out_dir = Path("outputs/runs/t1-colocate")
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for layout, members in LAYOUTS.items():
        idle = idle_vram_mib()
        servers = []
        try:
            for i, ((name, file), args) in enumerate(members):
                server = Server(
                    BIN, MODELS / file, [*COMMON, *args], 8081 + i, out_dir / f"{layout}-{name}.log"
                )
                if args == CPU:
                    server._require_gpu = lambda: None  # CPU on purpose
                load_s = server.start()
                servers.append((name, server, load_s))
            vram = idle_vram_mib() - idle
            row = {
                "layout": layout,
                "vram_mib": vram,
                "available_ram_mib": available_ram_mib(),
                "servers": [],
            }
            for name, server, load_s in servers:
                client = OpenAI(base_url=f"http://127.0.0.1:{server.port}/v1", api_key="none", timeout=900)
                calls = [
                    chat(
                        client,
                        name,
                        [{"role": "user", "content": SPEED_PROMPT}],
                        {"max_tokens": 256, "temperature": 0.6, "cache_prompt": False, **NO_THINK},
                    )
                    for _ in range(3)
                ]
                row["servers"].append(
                    {
                        "model": name,
                        "load_s": round(load_s, 1),
                        "rss_mib": rss_mib(server.proc.pid),
                        "ttft_s": statistics.median(c.ttft_s or 0 for c in calls),
                        "prefill_tps": statistics.median(
                            (c.timings or {}).get("prompt_per_second", 0) for c in calls
                        ),
                        "decode_tps": statistics.median(
                            (c.timings or {}).get("predicted_per_second", 0) for c in calls
                        ),
                        "errors": [c.error for c in calls if c.error],
                    }
                )
            results.append(row)
            print(json.dumps(row), flush=True)
        except (RuntimeError, TimeoutError) as e:
            results.append({"layout": layout, "error": str(e)})
            print(f"{layout}: {e}", flush=True)
        finally:
            for _, server, _ in reversed(servers):
                server.stop()
            time.sleep(3)
    (out_dir / "summary.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
