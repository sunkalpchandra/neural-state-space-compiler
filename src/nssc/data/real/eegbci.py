"""PhysioNet EEG Motor Movement/Imagery (EEGBCI) loader → :class:`TrajectoryDataset`.

Source: Schalk et al. (2004) BCI2000 / Goldberger et al. (2000) PhysioNet, 109 subjects,
64-channel EEG at 160 Hz, 14 runs per subject (see ``docs/datasets_real.md``).
Downloaded on demand with :func:`mne.datasets.eegbci.load_data`; ``mne`` is an optional
extra (``pip install nssc[eeg]``) and is imported lazily inside :func:`build_eegbci`.

Config schema (defaults in :data:`DEFAULTS`)::

    source: eegbci
    subjects: [1, 2, 3, 4]         # PhysioNet subject ids (1..109)
    runs: [3, 7, 11]               # run ids; 3,7,11 = real L/R fist, 6,10,14 = imagined hands/feet
    channels: null | int | [names] # null = all 64; int = first n (file order); list = names
    resample_hz: 64                # downsample from 160 Hz (null = keep 160)
    bandpass: [1.0, 30.0]          # FIR band-pass (mne); null = none
    segment_seconds: 8             # each run is cut into fixed-length segments (= trajectories)
    segment_stride_seconds: 8      # stride between segment starts (== length → no overlap)
    per_subject_standardize: true  # z-score per subject+channel with THAT subject's own data
    split: {by: subject, train_subjects: [1, 2], val_subjects: [3], test_subjects: [4]}
    cache_dir: data/cache/eegbci

Output ``x`` is ``(N_segments, T, C)`` float32, ``t`` in seconds, ``z_true=None`` (there is
no ground-truth latent). ``metadata['split_indices']`` holds the subject-level split which
:meth:`TrajectoryDataset.split` honours; per-segment ``subject_of_segment`` /
``run_of_segment`` are listed in ``metadata['per_traj_keys']`` so subsets stay aligned.
Processed datasets are cached as ``<cache_dir>/eegbci_<version>.npz`` keyed by the
``stable_hash`` of the resolved config.

Standardisation note: per-subject z-scoring uses only that subject's own recordings, so
no statistic from a test subject ever touches training data (and vice versa). The
downstream trainer additionally normalises with *train-split* statistics.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any

import numpy as np

from nssc.data.dataset import TrajectoryDataset
from nssc.data.splits import check_no_leakage
from nssc.utils.hashing import stable_hash

RAW_FS = 160.0
SOURCE = "eegbci"

DEFAULTS: dict[str, Any] = {
    "source": SOURCE,
    "subjects": [1, 2, 3, 4],
    "runs": [3, 7, 11],
    "channels": None,
    "resample_hz": 64,
    "bandpass": [1.0, 30.0],
    "segment_seconds": 8,
    "segment_stride_seconds": None,   # default: == segment_seconds
    "per_subject_standardize": True,
    "split": {"by": "subject", "train_subjects": [1, 2], "val_subjects": [3],
              "test_subjects": [4]},
    "cache_dir": "data/cache/eegbci",
    "montage": False,                 # set the standard_1005 montage (positions only; not needed)
}

_INSTALL_MSG = ("The EEGBCI loader needs the optional dependency `mne` "
                "(pip install mne  /  pip install 'nssc[eeg]').")


def resolve_eegbci_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Fill defaults and validate; the result is what gets hashed into ``metadata['version']``."""
    out = copy.deepcopy(DEFAULTS)
    out.update(copy.deepcopy(dict(cfg)))
    if out.get("source") != SOURCE:
        raise ValueError(f"expected source '{SOURCE}', got {out.get('source')!r}")
    out["subjects"] = [int(s) for s in out["subjects"]]
    out["runs"] = [int(r) for r in out["runs"]]
    if not out["subjects"] or not out["runs"]:
        raise ValueError("eegbci: 'subjects' and 'runs' must be non-empty")
    if any(s < 1 or s > 109 for s in out["subjects"]):
        raise ValueError("eegbci: subject ids are 1..109")
    if any(r < 1 or r > 14 for r in out["runs"]):
        raise ValueError("eegbci: run ids are 1..14")
    if out["segment_stride_seconds"] is None:
        out["segment_stride_seconds"] = out["segment_seconds"]
    if out["segment_seconds"] <= 0 or out["segment_stride_seconds"] <= 0:
        raise ValueError("eegbci: segment_seconds / segment_stride_seconds must be > 0")
    if out["resample_hz"] is not None and float(out["resample_hz"]) > RAW_FS:
        raise ValueError(f"eegbci: resample_hz must be <= {RAW_FS}")
    ch = out["channels"]
    if ch is not None and not isinstance(ch, int) and not isinstance(ch, (list, tuple)):
        raise ValueError("eegbci: channels must be null, an int or a list of names")
    if isinstance(ch, (list, tuple)):
        out["channels"] = [str(c) for c in ch]
    split = dict(out["split"] or {})
    if split.get("by", "subject") != "subject":
        raise ValueError("eegbci: only split.by == 'subject' is supported (no timestep leakage)")
    split["by"] = "subject"
    for k in ("train_subjects", "val_subjects", "test_subjects"):
        split[k] = [int(s) for s in split.get(k, [])]
    out["split"] = split
    validate_subject_split(out["subjects"], split)
    return out


def validate_subject_split(subjects: list[int], split: dict[str, Any]) -> None:
    """Raise if split subject sets overlap, are missing, or reference unloaded subjects."""
    tr, va, te = (set(split.get(k, [])) for k in ("train_subjects", "val_subjects",
                                                  "test_subjects"))
    if not tr:
        raise ValueError("eegbci split: train_subjects must be non-empty")
    if tr & va or tr & te or va & te:
        raise ValueError("eegbci split: a subject may appear in only one of train/val/test "
                         f"(train={sorted(tr)}, val={sorted(va)}, test={sorted(te)})")
    unknown = (tr | va | te) - set(subjects)
    if unknown:
        raise ValueError(f"eegbci split references subjects not in 'subjects': {sorted(unknown)}")


def subject_split_indices(subject_of_segment: list[int], split: dict[str, Any]
                          ) -> dict[str, list[int]]:
    """Segment indices per split from the per-segment subject list (subject-level, leak-free)."""
    subj = np.asarray(subject_of_segment)
    out: dict[str, list[int]] = {}
    for name in ("train", "val", "test"):
        wanted = set(split.get(f"{name}_subjects", []))
        out[name] = [int(i) for i in np.flatnonzero(np.isin(subj, list(wanted)))]
    check_no_leakage(out["train"], out["val"], out["test"])
    return out


def cache_path(cfg: dict[str, Any]) -> Path:
    """``<cache_dir>/eegbci_<version>.npz`` for a *resolved* config."""
    return Path(cfg["cache_dir"]) / f"eegbci_{stable_hash(cfg)}.npz"


# --------------------------------------------------------------------------- loading
def _import_mne():
    try:
        import mne  # noqa: PLC0415
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(_INSTALL_MSG) from e
    mne.set_log_level("ERROR")
    return mne


def _download(mne, subject: int, runs: list[int], raw_dir: Path) -> list[Path]:
    try:
        files = mne.datasets.eegbci.load_data(subject, runs, path=str(raw_dir),
                                              update_path=False, verbose=False)
    except Exception as e:  # noqa: BLE001 - network / IO errors of many kinds
        raise RuntimeError(
            f"eegbci: could not download subject {subject} runs {runs} into {raw_dir}. "
            "This loader needs network access to https://physionet.org/files/eegmmidb/1.0.0/ "
            "on first use (files are cached afterwards). Pre-populate the cache offline with "
            "mne.datasets.eegbci.load_data(...) or copy the EDF files into "
            f"{raw_dir}/MNE-eegbci-data/files/eegmmidb/1.0.0/S{subject:03d}/. "
            f"Underlying error: {type(e).__name__}: {e}") from e
    return [Path(f) for f in files]


def _load_run(mne, path: Path, cfg: dict[str, Any]) -> tuple[np.ndarray, list[str], float,
                                                              dict[str, int]]:
    """Read one EDF run → ``(data (T, C) float32, channel names, fs, annotation counts)``."""
    raw = mne.io.read_raw_edf(str(path), preload=True, verbose=False)
    mne.datasets.eegbci.standardize(raw)  # channel names → standard 10-05 form (e.g. 'C3')
    if cfg.get("montage"):
        raw.set_montage(mne.channels.make_standard_montage("standard_1005"),
                        on_missing="ignore", verbose=False)
    ch = cfg["channels"]
    if isinstance(ch, int):
        raw.pick(raw.ch_names[:ch])
    elif isinstance(ch, list):
        missing = [c for c in ch if c not in raw.ch_names]
        if missing:
            raise ValueError(f"eegbci: unknown channel names {missing}; "
                             f"available: {raw.ch_names}")
        raw.pick(ch)
    bp = cfg["bandpass"]
    if bp:
        raw.filter(float(bp[0]) if bp[0] is not None else None,
                   float(bp[1]) if bp[1] is not None else None, verbose=False)
    if cfg["resample_hz"] is not None and float(cfg["resample_hz"]) != raw.info["sfreq"]:
        raw.resample(float(cfg["resample_hz"]), verbose=False)
    counts: dict[str, int] = {}
    for d in raw.annotations.description:
        counts[str(d)] = counts.get(str(d), 0) + 1
    data = raw.get_data().T.astype(np.float32)  # (T, C), volts
    return data, list(raw.ch_names), float(raw.info["sfreq"]), counts


def _segment(data: np.ndarray, length: int, stride: int) -> np.ndarray:
    """``(T, C)`` → ``(n_seg, length, C)`` fixed-length windows (tail shorter than length dropped)."""
    n = (data.shape[0] - length) // stride + 1 if data.shape[0] >= length else 0
    if n <= 0:
        return np.zeros((0, length, data.shape[1]), dtype=data.dtype)
    return np.stack([data[i * stride:i * stride + length] for i in range(n)])


def build_eegbci(cfg: dict[str, Any], use_cache: bool = True) -> TrajectoryDataset:
    """Download (if needed), preprocess, segment and package EEGBCI as a TrajectoryDataset.

    Returns ``x (N_segments, T, C)`` float32 with subject-level ``metadata['split_indices']``.
    Raises ``RuntimeError`` with instructions when ``mne`` or the network is unavailable.
    """
    cfg = resolve_eegbci_config(cfg)
    version = stable_hash(cfg)
    cpath = cache_path(cfg)
    if use_cache and cpath.exists():
        ds = TrajectoryDataset.load(cpath)
        if ds.metadata.get("version") == version:
            ds.metadata["cache_hit"] = True
            return ds

    mne = _import_mne()
    t0 = time.perf_counter()
    raw_dir = Path(cfg["cache_dir"]) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    per_subject: list[tuple[int, int, np.ndarray]] = []  # (subject, run, (T, C))
    ch_names: list[str] | None = None
    fs: float | None = None
    annotations: dict[str, dict[str, int]] = {}
    for subject in cfg["subjects"]:
        files = _download(mne, subject, cfg["runs"], raw_dir)
        run_data: list[tuple[int, np.ndarray]] = []
        for run, path in zip(cfg["runs"], files):
            data, names, sf, counts = _load_run(mne, path, cfg)
            if ch_names is None:
                ch_names, fs = names, sf
            elif names != ch_names or sf != fs:
                raise RuntimeError(f"eegbci: inconsistent channels/fs in S{subject:03d}R{run:02d}")
            annotations[f"S{subject:03d}R{run:02d}"] = counts
            run_data.append((run, data))
        if cfg["per_subject_standardize"]:
            allx = np.concatenate([d for _, d in run_data], axis=0).astype(np.float64)
            mean, std = allx.mean(0), allx.std(0)
            std = np.where(std < 1e-12, 1.0, std)
            run_data = [(r, ((d - mean) / std).astype(np.float32)) for r, d in run_data]
        else:  # volts → microvolts so values are O(1..100) rather than 1e-5
            run_data = [(r, (d * 1e6).astype(np.float32)) for r, d in run_data]
        per_subject.extend((subject, r, d) for r, d in run_data)

    assert fs is not None and ch_names is not None
    length = int(round(cfg["segment_seconds"] * fs))
    stride = int(round(cfg["segment_stride_seconds"] * fs))
    segs, subj_of, run_of = [], [], []
    for subject, run, data in per_subject:
        s = _segment(data, length, stride)
        segs.append(s)
        subj_of += [subject] * len(s)
        run_of += [run] * len(s)
    x = np.concatenate(segs, axis=0) if segs else np.zeros((0, length, len(ch_names)), np.float32)
    if x.shape[0] == 0:
        raise RuntimeError("eegbci: no segments produced (segment_seconds longer than a run?)")
    if not np.all(np.isfinite(x)):
        raise FloatingPointError("eegbci: non-finite values after preprocessing")

    split_indices = subject_split_indices(subj_of, cfg["split"])
    t = np.arange(length, dtype=np.float64) / fs
    metadata = {
        "source": SOURCE, "subjects": cfg["subjects"], "runs": cfg["runs"], "fs": fs,
        "dt": 1.0 / fs, "channels": ch_names, "n_channels": len(ch_names),
        "segment_len": length, "segment_stride": stride,
        "subject_of_segment": subj_of, "run_of_segment": run_of,
        "per_traj_keys": ["subject_of_segment", "run_of_segment"],
        "annotations": annotations,          # per run: {'T0': n_rest, 'T1': n, 'T2': n}
        "n_segments_per_subject": {str(s): int(sum(1 for v in subj_of if v == s))
                                   for s in cfg["subjects"]},
        "split_indices": split_indices, "split_by": "subject",
        "per_subject_standardize": bool(cfg["per_subject_standardize"]),
        "config": cfg, "version": version, "build_time_s": time.perf_counter() - t0,
        "cache_hit": False,
    }
    ds = TrajectoryDataset(x=x, t=t, z_true=None, mask=None, metadata=metadata)
    if use_cache:
        ds.save(cpath)
    return ds
