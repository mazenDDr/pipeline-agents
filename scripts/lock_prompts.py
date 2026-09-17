"""Lock the prompt versions that runs actually used, from their call logs (on the Mac, after pulling runs).

    python scripts/lock_prompts.py outputs/runs/<run_id> [...]

Locking happens here rather than during a run because the GPU machine's copy of prompts/ is replaced on every
push. Each call log records the template refs and sha256 it used; a ref whose local file no longer has that
hash is reported and not locked (the run used a different version of the file than the one in the repository).
"""

import json
import sys
from pathlib import Path

from pipeline_agents.harness.registry import PromptRegistry


def main() -> None:
    registry = PromptRegistry("prompts")
    used: dict[str, str] = {}
    for run_dir in map(Path, sys.argv[1:]):
        for path in run_dir.rglob("calls.jsonl"):
            for line in path.read_text().splitlines():
                row = json.loads(line)
                for kind in ("system", "user"):
                    used[row["prompt_refs"][kind]] = row["prompt_sha256"][kind]
    problems = 0
    for ref, digest in sorted(used.items()):
        role, rest = ref.split("/")
        kind, version = rest.split(".v")
        template = registry.get(role, kind, int(version))  # type: ignore[arg-type]
        if template.sha256 != digest:
            print(f"NOT LOCKED {ref}: the run used different content than prompts/ has now")
            problems += 1
            continue
        registry.lock(template)
        print(f"locked {ref}")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
