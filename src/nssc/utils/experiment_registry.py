"""Experiment registry: append-only JSONL ledger of every run.

Each run receives a monotonically increasing ID ``EXP-0001``. Records are
never deleted; failed runs are kept with ``status='failed'`` (CLAUDE.md rule).

Concurrency: several processes (parallel compiles / benchmark suites) write the
same ledger. ``register`` therefore takes an exclusive ``flock`` on a sidecar
lock file while it reads the highest id and appends the new row, so ids cannot
collide. Plain ``update``/``complete``/``fail`` appends are single-line writes
(atomic enough on POSIX for lines < PIPE_BUF) and do not take the lock.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from nssc.utils.env import git_commit, hardware_info
from nssc.utils.io import append_jsonl, read_jsonl

DEFAULT_REGISTRY = Path("results/registry.jsonl")


@contextlib.contextmanager
def _exclusive(path: Path):
    """Cross-process exclusive lock on ``<path>.lock`` (no-op if the fs refuses)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(path.suffix + ".lock")
    fd = os.open(lock, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
        except OSError:  # pragma: no cover - exotic filesystems
            pass
        yield
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


@dataclass
class ExperimentRecord:
    experiment_id: str
    git_commit: str
    config_hash: str
    dataset: str
    model: str
    seed: int
    status: str = "running"  # running | completed | failed
    metrics: dict[str, Any] = field(default_factory=dict)
    checkpoint: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    param_count: int | None = None
    train_time_s: float | None = None
    hardware: dict[str, Any] = field(default_factory=hardware_info)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    tags: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ExperimentRegistry:
    def __init__(self, path: str | Path = DEFAULT_REGISTRY) -> None:
        self.path = Path(path)

    # ------------------------------------------------------------------ read
    def records(self) -> list[dict[str, Any]]:
        """Latest record per experiment_id (later lines override earlier)."""
        latest: dict[str, dict[str, Any]] = {}
        for r in read_jsonl(self.path):
            latest[r["experiment_id"]] = r
        return list(latest.values())

    def get(self, experiment_id: str) -> dict[str, Any] | None:
        for r in self.records():
            if r["experiment_id"] == experiment_id:
                return r
        return None

    def find(self, **filters: Any) -> list[dict[str, Any]]:
        out = []
        for r in self.records():
            if all(r.get(k) == v for k, v in filters.items()):
                out.append(r)
        return out

    def find_by_hash(self, config_hash: str, seed: int | None = None) -> list[dict[str, Any]]:
        out = [r for r in self.records() if r["config_hash"] == config_hash]
        if seed is not None:
            out = [r for r in out if r["seed"] == seed]
        return out

    def next_id(self) -> str:
        ids = [int(r["experiment_id"].split("-")[1]) for r in read_jsonl(self.path)]
        return f"EXP-{(max(ids) + 1 if ids else 1):04d}"

    # ----------------------------------------------------------------- write
    def register(self, *, config: dict[str, Any], config_hash: str, dataset: str, model: str,
                 seed: int, tags: list[str] | None = None, notes: str = "",
                 device: str | None = None) -> ExperimentRecord:
        """Allocate the next id and append the row under an exclusive lock.

        ``device`` (e.g. ``"cpu"``, ``"mps"``) is stored in ``hardware['device']`` — the
        *actual* device the run used, which ``hardware_info()`` alone cannot know.
        """
        with _exclusive(self.path):
            hw = hardware_info()
            if device:
                hw["device"] = str(device)
            rec = ExperimentRecord(
                experiment_id=self.next_id(),
                git_commit=git_commit(),
                config_hash=config_hash,
                dataset=dataset,
                model=model,
                seed=seed,
                config=config,
                tags=tags or [],
                notes=notes,
                hardware=hw,
            )
            append_jsonl(rec.to_dict(), self.path)
        return rec

    def update(self, rec: ExperimentRecord, **fields: Any) -> ExperimentRecord:
        for k, v in fields.items():
            setattr(rec, k, v)
        rec.updated_at = time.time()
        append_jsonl(rec.to_dict(), self.path)
        return rec

    def complete(self, rec: ExperimentRecord, metrics: dict[str, Any], **fields: Any) -> ExperimentRecord:
        return self.update(rec, status="completed", metrics=metrics, **fields)

    def fail(self, rec: ExperimentRecord, error: str) -> ExperimentRecord:
        return self.update(rec, status="failed", notes=(rec.notes + "\n" + error).strip())
