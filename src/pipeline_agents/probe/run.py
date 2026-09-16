"""Model probe (T1): start each candidate in llama-server, then measure speed, memory, plan JSON and code.

Run on the GPU machine:
    python -m pipeline_agents.probe.run --config configs/probe.yaml [--models a,b] [--code-samples 1]

Writes to outputs/runs/<run_id>/: calls.jsonl (every request with its rendered messages and raw
response), results.jsonl (one scored row per probe item), servers.jsonl (load time and memory per model),
server-<model>-<mode>.log. Re-running with the same --run-id skips the rows already in results.jsonl.
"""

import argparse
import json
import re
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from openai import OpenAI

from pipeline_agents.probe import code_tasks, plan_tasks
from pipeline_agents.probe.sandbox import run_script

CODE_SYSTEM = (
    "You write one complete Python 3.11 script that solves a data task. The script runs in a directory that "
    "contains the input files; read and write files by relative path. Available libraries: pandas 2.3, "
    "numpy, scikit-learn 1.8. There is no network access. Reply with only the script, in a single ```python "
    "code block."
)


# --- memory sampling -------------------------------------------------------------------------


@dataclass
class MemorySampler:
    """Polls GPU memory (nvidia-smi) and the server's resident memory while a model is loaded."""

    pid: int
    interval_s: float = 0.5
    peak_vram_mib: int = 0
    peak_rss_mib: int = 0
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join()

    def _loop(self) -> None:
        while not self._stop.is_set():
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
            ).stdout.strip()
            if out:
                self.peak_vram_mib = max(self.peak_vram_mib, int(out.splitlines()[0]))
            status = Path(f"/proc/{self.pid}/status")
            if status.exists():
                match = re.search(r"VmRSS:\s+(\d+) kB", status.read_text())
                if match:
                    self.peak_rss_mib = max(self.peak_rss_mib, int(match.group(1)) // 1024)
            self._stop.wait(self.interval_s)


def idle_vram_mib() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    ).stdout
    return int(out.strip().splitlines()[0])


# --- server ----------------------------------------------------------------------------------


class Server:
    def __init__(self, binary: str, model_path: Path, args: list[str], port: int, log_path: Path) -> None:
        self.cmd = [binary, "-m", str(model_path), "--port", str(port), "--host", "127.0.0.1", *args]
        self.port = port
        self.log_path = log_path
        self.proc: subprocess.Popen | None = None

    def start(self, timeout_s: float = 900) -> float:
        log = self.log_path.open("w")
        log.write(" ".join(self.cmd) + "\n")
        log.flush()
        start = time.perf_counter()
        self.proc = subprocess.Popen(self.cmd, stdout=log, stderr=subprocess.STDOUT)
        while time.perf_counter() - start < timeout_s:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited with {self.proc.returncode}; see {self.log_path}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2) as r:
                    if r.status == 200:
                        self._require_gpu()
                        return time.perf_counter() - start
            except OSError:
                pass
            time.sleep(1)
        self.stop()
        raise TimeoutError(f"llama-server not healthy after {timeout_s}s")

    def _require_gpu(self) -> None:
        # A CPU-only build still serves requests, just ~10x slower, which would silently corrupt every
        # speed number. Seen once already: the wrong llama.cpp build dir.
        if "no usable GPU found" in self.log_path.read_text():
            self.stop()
            raise RuntimeError(f"llama-server is running without a GPU; see {self.log_path}")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()


# --- one model call --------------------------------------------------------------------------


# Everything else (top_k, chat_template_kwargs, ...) is llama-server specific and goes in extra_body.
OPENAI_PARAMS = {"temperature", "top_p", "max_tokens", "seed", "response_format", "stop"}


@dataclass
class Call:
    content: str
    reasoning: str
    finish_reason: str | None
    ttft_s: float | None  # first token of any kind, reasoning included
    ttfc_s: float | None  # first answer (content) token
    total_s: float
    prompt_tokens: int | None
    completion_tokens: int | None
    timings: dict[str, Any] | None  # llama-server's own prompt/decode speeds
    error: str | None = None


def chat(client: OpenAI, model: str, messages: list[dict], params: dict[str, Any]) -> Call:
    start = time.perf_counter()
    content, reasoning = [], []
    ttft = ttfc = None
    finish = usage = timings = None
    try:
        known = {k: v for k, v in params.items() if k in OPENAI_PARAMS}
        extra = {k: v for k, v in params.items() if k not in OPENAI_PARAMS}
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
            extra_body=extra or None,
            **known,
        )
        for chunk in stream:
            chunk_extra = chunk.model_extra or {}
            if chunk_extra.get("timings"):
                timings = chunk_extra["timings"]
            if chunk.usage:
                usage = chunk.usage
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            piece_reasoning = (delta.model_extra or {}).get("reasoning_content") or ""
            now = time.perf_counter() - start
            if (piece_reasoning or delta.content) and ttft is None:
                ttft = now
            if delta.content and ttfc is None:
                ttfc = now
            reasoning.append(piece_reasoning)
            content.append(delta.content or "")
            finish = choice.finish_reason or finish
    except Exception as e:  # recorded, not raised: one bad call must not end a multi-hour probe
        return Call(
            "".join(content),
            "".join(reasoning),
            finish,
            ttft,
            ttfc,
            time.perf_counter() - start,
            None,
            None,
            timings,
            error=f"{type(e).__name__}: {e}",
        )
    return Call(
        "".join(content),
        "".join(reasoning),
        finish,
        ttft,
        ttfc,
        time.perf_counter() - start,
        usage.prompt_tokens if usage else None,
        usage.completion_tokens if usage else None,
        timings,
    )


# --- probes ----------------------------------------------------------------------------------


def extract_code(text: str) -> str:
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    return max(blocks, key=len) if blocks else text


def file_preview(workdir: Path, max_lines: int = 6) -> str:
    parts = []
    for path in sorted(workdir.iterdir()):
        lines = path.read_text().splitlines()
        parts.append(f"--- {path.name} ({len(lines)} lines) ---\n" + "\n".join(lines[:max_lines]))
    return "\n".join(parts)


SPEED_PROMPT = (
    "Below is a column profile of a wide dataset.\n"
    + "\n".join(
        f"col_{i:03d}: dtype={'float64' if i % 3 else 'object'}, nulls={i % 7 * 1.3:.1f}%, "
        f"unique={(i * 37) % 900 + 2}, example values=[{i * 1.5:.2f}, {i * 2.25:.2f}, {i * 3.125:.3f}], "
        f"note=column {i} was exported from the billing system and may contain sentinel values"
        for i in range(90)
    )
    + "\n\nList the five columns you would inspect first and why, in five short bullet points."
)


class Probe:
    def __init__(self, cfg: dict, run_dir: Path) -> None:
        self.cfg = cfg
        self.run_dir = run_dir
        self.done = self._load_done()

    def _load_done(self) -> set[str]:
        path = self.run_dir / "results.jsonl"
        if not path.exists():
            return set()
        return {json.loads(line)["key"] for line in path.read_text().splitlines() if line.strip()}

    def _append(self, name: str, row: dict) -> None:
        with (self.run_dir / name).open("a") as f:
            f.write(json.dumps(row) + "\n")

    def _call(self, client: OpenAI, ctx: dict, messages: list[dict], params: dict) -> Call:
        call = chat(client, "probe", messages, params)
        self._append("calls.jsonl", {**ctx, "messages": messages, "params": params, **asdict(call)})
        return call

    def _record(self, key: str, row: dict) -> None:
        self._append("results.jsonl", {"key": key, **row})
        self.done.add(key)

    def run_model(self, model: dict, mode: str, code_samples: int) -> None:
        mode_params = model["modes"][mode]
        tag = f"{model['name']}|{mode}"
        wanted = (
            [f"{tag}|speed|{i}" for i in range(self.cfg["speed_repeats"])]
            + [f"{tag}|plan|{c.id}|{s}" for c in plan_tasks.CASES for s in ("prompt", "enforced")]
            + [f"{tag}|code|{t.id}|{k}" for t in code_tasks.TASKS for k in range(code_samples)]
        )
        if all(k in self.done for k in wanted):
            print(f"skip {tag} (done)", flush=True)
            return

        server = Server(
            self.cfg["server_binary"],
            Path(self.cfg["model_dir"]) / model["file"],
            [*self.cfg["server_args"], *model.get("server_args", [])],
            self.cfg["port"],
            self.run_dir / f"server-{model['name']}-{mode}.log",
        )
        idle = idle_vram_mib()
        load_s = server.start()
        sampler = MemorySampler(server.proc.pid)
        sampler.start()
        client = OpenAI(base_url=f"http://127.0.0.1:{self.cfg['port']}/v1", api_key="none", timeout=1800)
        base = {**self.cfg["sampling"], **mode_params}
        print(f"{tag}: loaded in {load_s:.0f}s", flush=True)
        try:
            self._speed(client, tag, base)
            self._plans(client, tag, base)
            self._code(client, tag, base, code_samples)
        finally:
            sampler.stop()
            server.stop()
            self._append(
                "servers.jsonl",
                {
                    "model": model["name"],
                    "mode": mode,
                    "load_s": load_s,
                    "idle_vram_mib": idle,
                    "peak_vram_mib": sampler.peak_vram_mib,
                    "peak_rss_mib": sampler.peak_rss_mib,
                    "time": datetime.now().isoformat(timespec="seconds"),
                },
            )

    def _speed(self, client: OpenAI, tag: str, base: dict) -> None:
        for i in range(self.cfg["speed_repeats"]):
            key = f"{tag}|speed|{i}"
            if key in self.done:
                continue
            # Speed is measured with a fixed output length, whatever the mode's max_tokens.
            # cache_prompt off: otherwise repeats 2-3 reuse the KV cache and measure nothing.
            params = {**base, "max_tokens": 256, "cache_prompt": False}
            call = self._call(client, {"key": key}, [{"role": "user", "content": SPEED_PROMPT}], params)
            t = call.timings or {}
            self._record(
                key,
                {
                    "phase": "speed",
                    "ttft_s": call.ttft_s,
                    "total_s": call.total_s,
                    "prompt_tokens": call.prompt_tokens,
                    "completion_tokens": call.completion_tokens,
                    "prompt_tps": t.get("prompt_per_second"),
                    "decode_tps": t.get("predicted_per_second"),
                    "error": call.error,
                },
            )

    def _plans(self, client: OpenAI, tag: str, base: dict) -> None:
        for case in plan_tasks.CASES:
            for style in ("prompt", "enforced"):
                key = f"{tag}|plan|{case.id}|{style}"
                if key in self.done:
                    continue
                params = dict(base)
                if style == "enforced":
                    params["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {"name": "plan", "schema": plan_tasks.SCHEMA},
                    }
                messages = [
                    {"role": "system", "content": plan_tasks.SYSTEM},
                    {"role": "user", "content": plan_tasks.user_prompt(case, with_schema=style == "prompt")},
                ]
                call = self._call(client, {"key": key}, messages, params)
                verdict = plan_tasks.judge(case, call.content)
                self._record(
                    key,
                    {
                        "phase": "plan",
                        "style": style,
                        "case": case.id,
                        "valid": verdict.valid,
                        "sound": verdict.sound,
                        "detail": verdict.detail,
                        "finish_reason": call.finish_reason,
                        "total_s": call.total_s,
                        "completion_tokens": call.completion_tokens,
                        "reasoning_chars": len(call.reasoning),
                        "error": call.error,
                    },
                )

    def _code(self, client: OpenAI, tag: str, base: dict, samples: int) -> None:
        for task in code_tasks.TASKS:
            for k in range(samples):
                key = f"{tag}|code|{task.id}|{k}"
                if key in self.done:
                    continue
                with tempfile.TemporaryDirectory() as tmp:
                    work, hidden = Path(tmp) / "work", Path(tmp) / "hidden"
                    work.mkdir()
                    hidden.mkdir()
                    task.build(work, hidden, self.cfg["data_seed"])
                    messages = [
                        {"role": "system", "content": CODE_SYSTEM},
                        {"role": "user", "content": f"{task.prompt}\n\nFiles:\n{file_preview(work)}"},
                    ]
                    call = self._call(client, {"key": key}, messages, {**base, "seed": k})
                    code = extract_code(call.content)
                    run = run_script(code, work, timeout_s=self.cfg["code_timeout_s"])
                    check = (
                        task.check(work, hidden) if run.exit_code == 0 else code_tasks.CheckResult(False, "")
                    )
                    self._record(
                        key,
                        {
                            "phase": "code",
                            "task": task.id,
                            "sample": k,
                            "passed": check.passed,
                            "detail": check.detail or run.stderr[-300:],
                            "exit_code": run.exit_code,
                            "finish_reason": call.finish_reason,
                            "total_s": call.total_s,
                            "completion_tokens": call.completion_tokens,
                            "reasoning_chars": len(call.reasoning),
                            "error": call.error,
                        },
                    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/probe.yaml")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--models", default=None, help="comma-separated names; default all")
    parser.add_argument("--code-samples", type=int, default=1)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M") + "_probe"
    run_dir = Path("outputs/runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, run_dir / "config.yaml")

    names = set(args.models.split(",")) if args.models else None
    probe = Probe(cfg, run_dir)
    for model in cfg["models"]:
        if names and model["name"] not in names:
            continue
        if not (Path(cfg["model_dir"]) / model["file"]).exists():
            print(f"missing {model['file']}, skipping", flush=True)
            continue
        for mode in model["modes"]:
            try:
                probe.run_model(model, mode, args.code_samples)
            except (RuntimeError, TimeoutError) as e:
                print(f"{model['name']}|{mode}: server failed: {e}", flush=True)
                probe._append("servers.jsonl", {"model": model["name"], "mode": mode, "error": str(e)})
    print(f"done: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
