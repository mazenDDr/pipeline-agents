"""Versioned prompt registry (T3).

Prompts live in files, one directory per role:

    prompts/<role>/system.v<N>.md     the role's scope, output schema and constraints
    prompts/<role>/user.v<N>.j2       a Jinja2 template filled from the run state

A run pins the versions it uses, so a metric change can be traced to a prompt change. Once a version has
been used, it must never change: `prompts/lock.json` records the sha256 of every locked version, and
`check_locked` (run by the tests) fails if a locked file was edited. Change a prompt by adding a new version.
"""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Kind = Literal["system", "user"]
SUFFIX = {"system": "md", "user": "j2"}


@dataclass(frozen=True)
class PromptTemplate:
    role: str
    kind: Kind
    version: int
    text: str
    sha256: str

    @property
    def ref(self) -> str:
        return f"{self.role}/{self.kind}.v{self.version}"


class PromptRegistry:
    def __init__(self, root: Path | str = "prompts") -> None:
        self.root = Path(root)

    def path(self, role: str, kind: Kind, version: int) -> Path:
        return self.root / role / f"{kind}.v{version}.{SUFFIX[kind]}"

    def versions(self, role: str, kind: Kind) -> list[int]:
        pattern = re.compile(rf"{kind}\.v(\d+)\.{SUFFIX[kind]}$")
        found = (pattern.match(p.name) for p in (self.root / role).glob(f"{kind}.v*.{SUFFIX[kind]}"))
        return sorted(int(m.group(1)) for m in found if m)

    def latest(self, role: str, kind: Kind) -> int:
        versions = self.versions(role, kind)
        if not versions:
            raise FileNotFoundError(f"no {kind} prompt for role {role!r} under {self.root}")
        return versions[-1]

    def get(self, role: str, kind: Kind, version: int | None = None) -> PromptTemplate:
        version = self.latest(role, kind) if version is None else version
        path = self.path(role, kind, version)
        text = path.read_text()
        return PromptTemplate(role, kind, version, text, hashlib.sha256(text.encode()).hexdigest())

    # --- locking ---------------------------------------------------------------------------

    @property
    def lock_path(self) -> Path:
        return self.root / "lock.json"

    def locked(self) -> dict[str, str]:
        return json.loads(self.lock_path.read_text()) if self.lock_path.exists() else {}

    def lock(self, template: PromptTemplate) -> None:
        """Record a version as used. Locking the same content twice is a no-op; different content fails."""
        locked = self.locked()
        known = locked.get(template.ref)
        if known and known != template.sha256:
            raise ValueError(f"{template.ref} is locked with different content; add a new version instead")
        if not known:
            locked[template.ref] = template.sha256
            self.lock_path.write_text(json.dumps(dict(sorted(locked.items())), indent=2) + "\n")

    def check_locked(self) -> list[str]:
        """Refs whose file no longer matches the locked hash (or is missing)."""
        problems = []
        for ref, digest in self.locked().items():
            role, rest = ref.split("/")
            kind, version = rest.split(".v")
            path = self.path(role, kind, int(version))  # type: ignore[arg-type]
            if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                problems.append(ref)
        return problems
