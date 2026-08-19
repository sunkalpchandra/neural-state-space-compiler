"""Persistent, resumable search state (JSON file in the compile output directory).

Keyed by ``(stage, candidate_id, seed)``. If a search crashes after N runs, the
next invocation skips those N and continues.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from nssc.utils.io import load_json, save_json


class SearchState:
    """Resumable search state.

    Keys are namespaced by run protocol (``nssc.experiment.PROTOCOL_VERSION``) so a search resumed
    after a semantic change does **not** silently reuse runs trained under the old semantics; the
    old entries stay in the file as evidence (F-010).
    """

    def __init__(self, path: str | Path, namespace: str | None = None) -> None:
        self.path = Path(path)
        if namespace is None:
            from nssc.experiment import PROTOCOL_VERSION

            namespace = f"p{PROTOCOL_VERSION}"
        self.namespace = namespace
        self.data: dict[str, Any] = {"runs": {}, "stages": {}, "meta": {}, "created": time.time()}
        if self.path.exists():
            self.data = load_json(self.path)

    def key(self, stage: str, cand_id: str, seed: int) -> str:
        return f"{self.namespace}|{stage}|{cand_id}|{seed}"

    def has(self, stage: str, cand_id: str, seed: int) -> bool:
        r = self.data["runs"].get(self.key(stage, cand_id, seed))
        return bool(r) and r.get("status") in ("completed", "failed")

    def get(self, stage: str, cand_id: str, seed: int) -> dict[str, Any] | None:
        return self.data["runs"].get(self.key(stage, cand_id, seed))

    def put(self, stage: str, cand_id: str, seed: int, result: dict[str, Any]) -> None:
        self.data["runs"][self.key(stage, cand_id, seed)] = result
        self.save()

    def stage_results(self, stage: str) -> dict[str, list[dict[str, Any]]]:
        out: dict[str, list[dict[str, Any]]] = {}
        for k, v in self.data["runs"].items():
            parts = k.split("|")
            ns, st, cid = (parts[0], parts[1], parts[2]) if len(parts) == 4 else ("p1", parts[0], parts[1])
            if st == stage and ns == self.namespace:
                out.setdefault(cid, []).append(v)
        return out

    def set_stage(self, stage: str, info: dict[str, Any]) -> None:
        self.data["stages"][f"{self.namespace}|{stage}"] = info
        self.data["stages"][stage] = info  # convenience alias for the current namespace
        self.save()

    def get_stage(self, stage: str) -> dict[str, Any] | None:
        return self.data["stages"].get(f"{self.namespace}|{stage}")

    def save(self) -> None:
        self.data["updated"] = time.time()
        save_json(self.data, self.path)

    @property
    def n_completed(self) -> int:
        return sum(1 for k, r in self.data["runs"].items()
                   if r.get("status") == "completed" and k.startswith(self.namespace + "|"))

    def namespaced_runs(self) -> dict[str, Any]:
        return {k: v for k, v in self.data["runs"].items() if k.startswith(self.namespace + "|")}
