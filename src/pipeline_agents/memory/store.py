"""Memory storage (T8): items in JSONL with their embeddings, and the embedders.

Three stores are kept apart on purpose (semantic, procedural, episodic): each has its own writer, retrieval
rule and leakage rule, and the ablation switches them on and off independently.

Embedding choice, measured on 8 hand-labelled (paraphrased query, skill) pairs (scratch probe, 2026-09-17):
bge-small-en-v1.5 without the query instruction top-1 0.88, mean rank 1.38; with the instruction 0.75;
TF-IDF 0.25. Similarity margins were small (+0.03), so retrieval uses a fixed top-k, never a threshold.
"""

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
from pydantic import BaseModel, Field

StoreName = Literal["semantic", "procedural", "episodic"]


class MemoryItem(BaseModel):
    id: str
    store: StoreName
    key: str  # semantic: dataset fingerprint; procedural: step kind; episodic: task id
    text: str  # what goes into a prompt
    source_run: str
    source_task: str
    source_split: Literal["dev", "test", "user"]
    created: str = Field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    vector: list[float] = []


def item_id(store: str, key: str, text: str) -> str:
    return hashlib.sha256(f"{store}|{key}|{text}".encode()).hexdigest()[:16]


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> np.ndarray: ...


class HashEmbedder:
    """Deterministic bag-of-words hashing, for tests on machines without the model."""

    name = "hash-512"

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), 512))
        for i, text in enumerate(texts):
            for token in re.findall(r"[a-z0-9_]+", text.lower()):
                out[i, int(hashlib.md5(token.encode()).hexdigest(), 16) % 512] += 1
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-9)


class SentenceEmbedder:
    """bge-small-en-v1.5 on the CPU, so it never competes with the served models for VRAM. Loaded lazily."""

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5") -> None:
        self.name = model
        self._model = None

    def embed(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.name, device="cpu")
        return np.asarray(self._model.encode(texts, normalize_embeddings=True))


class JsonlStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def items(self) -> list[MemoryItem]:
        if not self.path.exists():
            return []
        return [
            MemoryItem.model_validate_json(line)
            for line in self.path.read_text().splitlines()
            if line.strip()
        ]

    def add(self, items: list[MemoryItem], embedder: Embedder) -> int:
        """Append items not already stored (same id). Returns how many were new."""
        known = {i.id for i in self.items()}
        new = [i for i in items if i.id not in known]
        if not new:
            return 0
        vectors = embedder.embed([i.text for i in new])
        with self.path.open("a") as f:
            for item, vector in zip(new, vectors, strict=True):
                f.write(
                    item.model_copy(update={"vector": [round(float(v), 6) for v in vector]}).model_dump_json()
                    + "\n"
                )
        return len(new)

    def search(
        self, query: str, embedder: Embedder, k: int, where=lambda item: True
    ) -> list[tuple[MemoryItem, float]]:
        candidates = [i for i in self.items() if where(i) and i.vector]
        if not candidates or k <= 0:
            return []
        q = embedder.embed([query])[0]
        scores = np.asarray([i.vector for i in candidates]) @ q
        order = np.argsort(-scores)[:k]
        return [(candidates[j], float(scores[j])) for j in order]

    def digest(self) -> dict[str, object]:
        """Which version of the store a run used: item count and a hash of the ids."""
        ids = sorted(i.id for i in self.items())
        return {"items": len(ids), "sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]}
