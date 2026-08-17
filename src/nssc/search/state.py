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
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.data: dict[str, Any] = {"runs": {}, "stages": {}, "meta": {}, "created": time.time()}
        if self.path.exists():
            self.data = load_json(self.path)

    @staticmethod
    def key(stage: str, cand_id: str, seed: int) -> str:
        return f"{stage}|{cand_id}|{seed}"

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
            st, cid, _ = k.split("|")
            if st == stage:
                out.setdefault(cid, []).append(v)
        return out

    def set_stage(self, stage: str, info: dict[str, Any]) -> None:
        self.data["stages"][stage] = info
        self.save()

    def get_stage(self, stage: str) -> dict[str, Any] | None:
        return self.data["stages"].get(stage)

    def save(self) -> None:
        self.data["updated"] = time.time()
        save_json(self.data, self.path)

    @property
    def n_completed(self) -> int:
        return sum(1 for r in self.data["runs"].values() if r.get("status") == "completed")
