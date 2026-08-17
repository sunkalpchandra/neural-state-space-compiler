"""nssc dashboard — FastAPI backend for the interactive dynamical-system explorer.

Run with ``nssc dashboard`` or ``python -m dashboard.app``. Sources are completed
runs in the experiment registries (``REGISTRY_PATHS``) whose checkpoint exists on
disk, plus compile directories under ``COMPILE_ROOT`` holding a
``compile_report.json``. Exactly one source is *loaded* at a time (model + test
split, cached per source key); the GET routes operate on the loaded source.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from nssc.experiment import prepare_data
from nssc.stability.spectral import jacobian_spectrum
from nssc.training.checkpoint import load_checkpoint
from nssc.uncertainty.rollout import probabilistic_rollout
from nssc.utils.experiment_registry import ExperimentRegistry
from nssc.utils.io import load_json

ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"

#: registries scanned by /api/sources (tests monkeypatch this list)
REGISTRY_PATHS: list[Path] = [Path("results/registry.jsonl"), Path("results/registry_smoke.jsonl")]
#: root of compile output directories
COMPILE_ROOT: Path = Path("results/compile")

MAX_PAYLOAD_DIMS = 8
MAX_SPECTRUM_POINTS = 128


# --------------------------------------------------------------------------- state
@dataclass
class Loaded:
    key: str
    model: Any
    meta: dict[str, Any]
    x: np.ndarray                    # (N, T, D) normalised test split
    t: np.ndarray                    # (T,)
    z_true: np.ndarray | None
    dt: float
    experiment_id: str
    model_name: str
    dataset: str
    git_commit: str
    metrics: dict[str, Any] = field(default_factory=dict)
    compile_report: dict[str, Any] | None = None
    checkpoint: str = ""
    _z_cache: dict[int, np.ndarray] = field(default_factory=dict)

    # -- helpers ------------------------------------------------------------
    @property
    def n_traj(self) -> int:
        return int(self.x.shape[0])

    @property
    def T(self) -> int:
        return int(self.x.shape[1])

    @property
    def obs_dim(self) -> int:
        return int(self.x.shape[2])

    @property
    def latent_dim(self) -> int:
        return int(self.model.latent_dim)

    def check_idx(self, idx: int) -> int:
        if not 0 <= idx < self.n_traj:
            raise HTTPException(400, f"idx must be in [0, {self.n_traj}), got {idx}")
        return idx

    @torch.no_grad()
    def latents(self, idx: int) -> np.ndarray:
        """Encoded latent trajectory z_{1:T} (T, d) for test trajectory ``idx``."""
        idx = self.check_idx(idx)
        if idx not in self._z_cache:
            xt = torch.from_numpy(self.x[idx : idx + 1])
            self._z_cache[idx] = self.model.encode(xt)[0].cpu().numpy()
        return self._z_cache[idx]

    def clamp_window(self, context: int, horizon: int) -> tuple[int, int]:
        context = int(max(1, min(context, self.T - 1)))
        horizon = int(max(1, min(horizon, self.T - context)))
        return context, horizon


_CACHE: dict[str, Loaded] = {}
_STATE: dict[str, Loaded | None] = {"current": None}


def _current() -> Loaded:
    cur = _STATE["current"]
    if cur is None:
        raise HTTPException(409, "no source loaded — POST /api/load first")
    return cur


def _finite(v: Any) -> Any:
    """Recursively replace NaN/inf by None so JSON stays valid."""
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, dict):
        return {k: _finite(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_finite(x) for x in v]
    if isinstance(v, np.ndarray):
        return _finite(v.tolist())
    if isinstance(v, (np.floating, np.integer)):
        return _finite(float(v))
    return v


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (Path.cwd() / p)


# ------------------------------------------------------------------ discovery
def _registry_records() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rp in REGISTRY_PATHS:
        for r in ExperimentRegistry(rp).records():
            if r.get("status") != "completed" or not r.get("checkpoint"):
                continue
            ck = _resolve(r["checkpoint"])
            if not (ck / "model.pt").exists():
                continue
            r = dict(r)
            r["_registry"] = str(rp)
            out.append(r)
    return out


_METRIC_SUBSET = ("val/recon/nrmse", "val/teacher_forced/nrmse", "val/recursive/nrmse_mean",
                  "test/recon/nrmse", "test/teacher_forced/nrmse", "test/recursive/nrmse@10",
                  "test/recursive/nrmse_mean", "test/stability/rho_max", "test/stability/verdict",
                  "test/params/total", "test/latent_dim")


def list_sources() -> dict[str, Any]:
    experiments = []
    recs = _registry_records()
    counts: dict[str, int] = {}
    for r in recs:
        counts[r["experiment_id"]] = counts.get(r["experiment_id"], 0) + 1
    for r in recs:
        m = r.get("metrics", {}) or {}
        eid = r["experiment_id"]
        # ids may collide across registries (e.g. registry.jsonl vs registry_smoke.jsonl)
        load_id = eid if counts[eid] == 1 else f"{eid}@{r['_registry']}"
        experiments.append({
            "id": eid, "load_id": load_id, "model": r.get("model"), "dataset": r.get("dataset"),
            "checkpoint": r.get("checkpoint"), "seed": r.get("seed"), "tags": r.get("tags", []),
            "params": r.get("param_count"), "registry": r["_registry"],
            "metrics": {k: m[k] for k in _METRIC_SUBSET if k in m},
        })
    compiles = []
    if COMPILE_ROOT.exists():
        for rep in sorted(COMPILE_ROOT.glob("*/compile_report.json")):
            try:
                j = load_json(rep)
            except Exception:  # noqa: BLE001
                continue
            ck = j.get("checkpoint")
            compiles.append({
                "dir": str(rep.parent), "selected": j.get("selected"),
                "dataset": (j.get("dataset") or {}).get("system"),
                "checkpoint": ck, "has_checkpoint": bool(ck and (_resolve(ck) / "model.pt").exists()),
                "n_runs": j.get("n_runs"),
            })
    return _finite({"experiments": experiments, "compiles": compiles})


# ---------------------------------------------------------------------- loading
def _load_from_checkpoint(key: str, ckpt: str, *, experiment_id: str, model_name: str,
                          dataset_name: str, git_commit: str, metrics: dict[str, Any],
                          dataset_cfg: dict[str, Any] | None,
                          compile_report: dict[str, Any] | None = None) -> Loaded:
    model, meta = load_checkpoint(_resolve(ckpt))
    model.eval()
    # the checkpoint's own metadata records the exact dataset config it was trained on
    dcfg = meta.get("dataset") or dataset_cfg
    if not dcfg:
        raise HTTPException(500, f"checkpoint {ckpt} has no dataset config in metadata")
    splits, _stats, raw = prepare_data(dcfg, seed=None)
    test = splits.get("test") or splits.get("val") or splits["train"]
    dt = float(raw.metadata.get("dt", dcfg.get("dt", 1.0)) or 1.0)
    return Loaded(key=key, model=model, meta=meta, x=np.asarray(test.x, dtype=np.float32),
                  t=np.asarray(test.t, dtype=np.float64), z_true=test.z_true, dt=dt,
                  experiment_id=experiment_id, model_name=model_name, dataset=dataset_name,
                  git_commit=git_commit, metrics=metrics, compile_report=compile_report,
                  checkpoint=ckpt)


def load_source(experiment_id: str | None = None, compile_dir: str | None = None) -> Loaded:
    if experiment_id:
        key = f"exp:{experiment_id}"
        if key not in _CACHE:
            eid, _, reg = experiment_id.partition("@")
            recs = [r for r in _registry_records() if r["experiment_id"] == eid
                    and (not reg or r["_registry"] == reg)]
            if not recs:
                raise HTTPException(404, f"no completed experiment {experiment_id} with checkpoint")
            r = recs[-1]
            _CACHE[key] = _load_from_checkpoint(
                key, r["checkpoint"], experiment_id=eid, model_name=r.get("model", "?"),
                dataset_name=r.get("dataset", "?"), git_commit=r.get("git_commit", ""),
                metrics=r.get("metrics", {}) or {}, dataset_cfg=(r.get("config") or {}).get("dataset"))
    elif compile_dir:
        key = f"compile:{compile_dir}"
        if key not in _CACHE:
            rep_path = _resolve(compile_dir) / "compile_report.json"
            if not rep_path.exists():
                raise HTTPException(404, f"no compile_report.json in {compile_dir}")
            rep = load_json(rep_path)
            ck = rep.get("checkpoint")
            if not ck or not (_resolve(ck) / "model.pt").exists():
                raise HTTPException(404, f"compile report in {compile_dir} has no checkpoint on disk")
            sel = rep.get("selected", {})
            # git commit lives on the registry record that produced the checkpoint
            same = [r for r in _registry_records() if r.get("checkpoint") == ck]
            commit = same[-1].get("git_commit", "") if same else ""
            mname = f"{sel.get('encoder', '?')}+{sel.get('dynamics', '?')}@d{sel.get('latent_dim', '?')}"
            _CACHE[key] = _load_from_checkpoint(
                key, ck, experiment_id=Path(compile_dir).name, model_name=mname,
                dataset_name=(rep.get("dataset") or {}).get("system", "?"),
                git_commit=commit,
                metrics=rep.get("selected_metrics", {}) or {}, dataset_cfg=rep.get("dataset"),
                compile_report=rep)
    else:
        raise HTTPException(400, "provide experiment_id or compile_dir")
    _STATE["current"] = _CACHE[key]
    return _CACHE[key]


def summary(L: Loaded) -> dict[str, Any]:
    params = L.model.num_parameters()
    return _finite({
        "key": L.key, "experiment_id": L.experiment_id, "model_name": L.model_name,
        "dataset": L.dataset, "git_commit": L.git_commit, "checkpoint": L.checkpoint,
        "obs_dim": L.obs_dim, "latent_dim": L.latent_dim, "n_traj": L.n_traj, "T": L.T,
        "params": params, "dt": L.dt, "has_z_true": L.z_true is not None,
        "true_latent_dim": None if L.z_true is None else int(L.z_true.shape[-1]),
        "is_stochastic": bool(getattr(L.model.dynamics, "is_stochastic", False)),
        "dynamics": type(L.model.dynamics).__name__, "encoder": type(L.model.encoder).__name__,
        "decoder": type(L.model.decoder).__name__,
        "dims": [f"x{i}" for i in range(L.obs_dim)],
        "latent_labels": [f"z{i + 1}" for i in range(L.latent_dim)],
        "metrics": L.metrics, "has_compile_report": L.compile_report is not None,
    })


# ------------------------------------------------------------------- analyses
def _dim_subset(D: int, k: int = MAX_PAYLOAD_DIMS) -> list[int]:
    if D <= k:
        return list(range(D))
    return [int(i) for i in np.linspace(0, D - 1, k).round()]


@torch.no_grad()
def trajectory(L: Loaded, idx: int, context: int, horizon: int, n_samples: int = 16
               ) -> dict[str, Any]:
    idx = L.check_idx(idx)
    context, horizon = L.clamp_window(context, horizon)
    x = L.x[idx]                                          # (T, D)
    z = L.latents(idx)                                    # (T, d)
    xt = torch.from_numpy(x[None])
    x_hat, z_hat = L.model.rollout(xt[:, :context], horizon)
    x_hat, z_hat = x_hat[0].cpu().numpy(), z_hat[0].cpu().numpy()
    truth = x[context : context + horizon]
    err = np.sqrt(((x_hat - truth) ** 2).mean(axis=1))    # (H,)
    scale = float(np.sqrt((x**2).mean())) or 1.0          # data normalised → ≈1
    nrmse = err / scale
    std = None
    method = None
    try:
        pr = probabilistic_rollout(L.model, xt[:, :context], horizon, n_samples=n_samples)
        std = pr["std"][0].cpu().numpy()
        method = pr["method"]
    except Exception:  # noqa: BLE001 — envelope is optional
        std = None
    dims = _dim_subset(L.obs_dim)
    out = {
        "idx": idx, "context": context, "horizon": horizon, "T": L.T, "dt": L.dt,
        "dims": dims, "dim_labels": [f"x{i}" for i in dims],
        "t": L.t.tolist(),
        "x": x[:, dims].tolist(), "z": z.tolist(),
        "x_hat": x_hat[:, dims].tolist(), "z_hat": z_hat.tolist(),
        "std": None if std is None else std[:, dims].tolist(), "std_method": method,
        "nrmse": nrmse.tolist(),
        "z_true": None if L.z_true is None else L.z_true[idx].tolist(),
    }
    return _finite(out)


@torch.no_grad()
def field(L: Loaded, dims: tuple[int, int], grid: int, idx: int = 0, pad: float = 0.15
          ) -> dict[str, Any]:
    d = L.latent_dim
    i, j = dims
    if not (0 <= i < d and 0 <= j < d):
        raise HTTPException(400, f"dims must be < latent_dim={d}")
    grid = int(max(5, min(grid, 61)))
    z = L.latents(idx)                                    # (T, d)
    centre = z.mean(axis=0)
    lo, hi = z.min(axis=0), z.max(axis=0)
    span = np.maximum(hi - lo, 1e-3)
    gx = np.linspace(lo[i] - pad * span[i], hi[i] + pad * span[i], grid)
    gy = np.linspace(lo[j] - pad * span[j], hi[j] + pad * span[j], grid) if j != i else gx
    X, Y = np.meshgrid(gx, gy)
    pts = np.tile(centre, (grid * grid, 1)).astype(np.float32)
    pts[:, i] = X.ravel()
    pts[:, j] = Y.ravel()
    zt = torch.from_numpy(pts)
    nxt = L.model.dynamics.step(zt).cpu().numpy()
    disp = nxt - pts
    speed = np.linalg.norm(disp[:, [i, j]], axis=1)
    return _finite({
        "dims": [i, j], "grid": grid, "x": gx.tolist(), "y": gy.tolist(),
        "u": disp[:, i].reshape(grid, grid).tolist(), "v": disp[:, j].reshape(grid, grid).tolist(),
        "speed": speed.reshape(grid, grid).tolist(),
        "traj": {"x": z[:, i].tolist(), "y": z[:, j].tolist()},
        "centre": centre.tolist(),
    })


def stability(L: Loaded, idx: int, max_points: int = MAX_SPECTRUM_POINTS) -> dict[str, Any]:
    z = torch.from_numpy(L.latents(idx))
    s = jacobian_spectrum(L.model.dynamics, z, max_points=max_points)
    eig = s["eigvals"].cpu().numpy()                      # (M, d) complex
    rho = s["spectral_radius"].cpu().numpy()
    M = eig.shape[0]
    tpos = np.linspace(0, L.T - 1, M).round().astype(int)
    m = L.metrics or {}
    pref = "test/" if any(k.startswith("test/") for k in m) else "val/"
    info = {k: m.get(pref + "stability/" + k) for k in ("rho_max", "verdict", "lyapunov_max",
                                                        "frac_blowup", "instability_score")}
    return _finite({
        "idx": idx, "n_points": int(M), "t_index": tpos.tolist(),
        "real": eig.real.tolist(), "imag": eig.imag.tolist(),
        "spectral_radius": rho.tolist(),
        "rho_max_local": float(rho.max()), "rho_mean_local": float(rho.mean()),
        "frac_expanding": float((rho > 1).mean()),
        "metrics": info, "metrics_split": pref.rstrip("/"),
    })


@torch.no_grad()
def counterfactual(L: Loaded, idx: int, context: int, horizon: int, z0: list[float]
                   ) -> dict[str, Any]:
    idx = L.check_idx(idx)
    context, horizon = L.clamp_window(context, horizon)
    if len(z0) != L.latent_dim:
        raise HTTPException(400, f"z0 must have {L.latent_dim} entries, got {len(z0)}")
    z = L.latents(idx)
    z_orig0 = z[context - 1]
    z_cf0 = np.asarray(z0, dtype=np.float32)
    zz = torch.from_numpy(np.stack([z_orig0, z_cf0]))
    x_hat, z_hat = L.model.rollout_from_latent(zz, horizon)
    x_hat, z_hat = x_hat.cpu().numpy(), z_hat.cpu().numpy()
    dims = _dim_subset(L.obs_dim)
    truth = L.x[idx, context : context + horizon]
    dz = float(np.linalg.norm(z_cf0 - z_orig0))
    dx = np.sqrt(((x_hat[1] - x_hat[0]) ** 2).mean(axis=1))
    return _finite({
        "idx": idx, "context": context, "horizon": horizon, "dims": dims,
        "dim_labels": [f"x{i}" for i in dims], "t": L.t.tolist(),
        "z0_original": z_orig0.tolist(), "z0": z_cf0.tolist(), "delta_z0": dz,
        "x_hat_original": x_hat[0][:, dims].tolist(), "x_hat": x_hat[1][:, dims].tolist(),
        "z_hat_original": z_hat[0].tolist(), "z_hat": z_hat[1].tolist(),
        "truth": truth[:, dims].tolist(), "divergence": dx.tolist(),
    })


# -------------------------------------------------------------------- FastAPI
class LoadRequest(BaseModel):
    experiment_id: str | None = None
    compile_dir: str | None = None


class CounterfactualRequest(BaseModel):
    idx: int = 0
    context: int = 20
    horizon: int = 50
    z0: list[float]


app = FastAPI(title="nssc dashboard", version="0.1")


@app.get("/api/sources")
def api_sources() -> JSONResponse:
    return JSONResponse(list_sources())


@app.post("/api/load")
def api_load(req: LoadRequest) -> JSONResponse:
    L = load_source(req.experiment_id, req.compile_dir)
    return JSONResponse(summary(L))


@app.get("/api/summary")
def api_summary() -> JSONResponse:
    return JSONResponse(summary(_current()))


@app.get("/api/trajectory")
def api_trajectory(idx: int = 0, context: int = 20, horizon: int = 50) -> JSONResponse:
    return JSONResponse(trajectory(_current(), idx, context, horizon))


@app.get("/api/field")
def api_field(dims: str = "0,1", grid: int = 21, idx: int = 0) -> JSONResponse:
    try:
        i, j = (int(v) for v in dims.split(","))
    except ValueError as e:
        raise HTTPException(400, "dims must be 'i,j'") from e
    return JSONResponse(field(_current(), (i, j), grid, idx))


@app.get("/api/stability")
def api_stability(idx: int = 0) -> JSONResponse:
    return JSONResponse(stability(_current(), idx))


@app.get("/api/compile_report")
def api_compile_report() -> JSONResponse:
    cur = _STATE["current"]
    return JSONResponse(_finite(cur.compile_report) if cur is not None else None)


@app.post("/api/counterfactual")
def api_counterfactual(req: CounterfactualRequest) -> JSONResponse:
    return JSONResponse(counterfactual(_current(), req.idx, req.context, req.horizon, req.z0))


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def serve(host: str = "127.0.0.1", port: int = 8050) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="nssc dashboard")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8050)
    a = ap.parse_args()
    serve(a.host, a.port)
