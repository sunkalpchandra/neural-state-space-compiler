# Real-world datasets (Tier 3)

Real data enter `nssc` through `nssc.data.real` (`source: <name>` in a dataset YAML).
The builder (`nssc.data.builder.build_dataset`) dispatches any config with a `source`
key to `nssc.data.real.build_real_dataset`; synthetic configs (`system: ...`) are untouched.
Heavy dependencies (`mne`) are imported lazily, only when a real source is built.

| source            | status          | module                          |
|-------------------|-----------------|---------------------------------|
| `eegbci`          | implemented     | `nssc/data/real/eegbci.py`      |
| `motion_cmu_mocap`| **stub** (raises `NotImplementedError`; see module docstring for the plan) | `nssc/data/real/motion.py` |

We deliberately do not ship synthetic stand-ins for unimplemented real sources.

## EEGBCI — PhysioNet EEG Motor Movement/Imagery

**What it is.** 109 volunteers, 64-channel scalp EEG (10-10 system, BCI2000), 160 Hz,
14 runs each: 2 baselines (eyes open/closed) and 12 task runs — three repetitions of
(3) open/close left or right fist, (4) imagine opening/closing left or right fist,
(5) open/close both fists or both feet, (6) imagine both fists or both feet.
Run ids: real fist movement = 3, 7, 11; imagined fist = 4, 8, 12; real fists/feet = 5, 9, 13;
imagined fists/feet = 6, 10, 14. Each task run lasts ~2 min (~125 s). Annotations
`T0` (rest), `T1`, `T2` (task onsets, ~4 s each) are kept in
`metadata['annotations']` per `S###R##` as counts only; we do not use labels.

**Access / licence.** Public, PhysioNet open-data (ODC-BY 1.0). Downloaded on first use
via `mne.datasets.eegbci.load_data(subject, runs, path=<cache_dir>/raw)` from
`https://physionet.org/files/eegmmidb/1.0.0/` (~1.5 MB EDF per subject-run) and cached.

**Citation (required by PhysioNet):**

* Schalk G., McFarland D.J., Hinterberger T., Birbaumer N., Wolpaw J.R. *BCI2000: A
  General-Purpose Brain-Computer Interface (BCI) System.* IEEE TBME 51(6):1034-1043, 2004.
* Goldberger A.L. et al. *PhysioBank, PhysioToolkit, and PhysioNet: Components of a New
  Research Resource for Complex Physiologic Signals.* Circulation 101(23):e215-e220, 2000.
* Loader: MNE-Python (Gramfort et al. 2013).

**Install:** `pip install mne` (or `pip install -e '.[eeg]'`). Without `mne` or without
network access on first build, `build_dataset` raises `RuntimeError` with instructions
(pre-populate `<cache_dir>/raw/MNE-eegbci-data/files/eegmmidb/1.0.0/S###/` offline).

### Preprocessing (in this order, per subject-run)

1. `mne.io.read_raw_edf(preload=True)`; `mne.datasets.eegbci.standardize` (channel names → `FC5`, `C3`, ...).
2. Channel selection: `channels: null` (all 64) | `int` (first *n* in file order) | list of names.
3. Band-pass FIR `bandpass: [lo, hi]` (mne defaults; default 1–30 Hz). Keep `hi` < resampled Nyquist.
4. Resample to `resample_hz` (default 64 Hz; raw is 160 Hz).
5. **Per-subject standardisation** (`per_subject_standardize: true`): per-channel z-score
   using the mean/std over *all loaded runs of that subject only*. No cross-subject
   statistics are used, so no test-subject statistic ever influences training data. If
   disabled, values are converted volts → microvolts. The trainer additionally normalises
   with **train-split** statistics (`prepare_data`).
6. Segmentation: each run is cut into fixed windows of `segment_seconds` with stride
   `segment_stride_seconds` (default equal → non-overlapping). Each segment is one
   trajectory: `x` is `(N_segments, T, C)` float32, `t` in seconds, `z_true = None`.

`metadata`: `source, subjects, runs, fs, dt, channels, segment_len, subject_of_segment,
run_of_segment, annotations, n_segments_per_subject, split_indices, config, version`
(`version = stable_hash(resolved config)`), plus `cache_hit`. The processed dataset is
cached at `<cache_dir>/eegbci_<version>.npz`; any change to the config produces a new
version/cache file. Metadata lists `per_traj_keys` so `TrajectoryDataset.subset/split`
keep the per-segment subject/run lists aligned.

### Split protocol — subject-level, never segment-level

`split: {by: subject, train_subjects: [...], val_subjects: [...], test_subjects: [...]}`.
Segments from a subject go to exactly one split (validated: overlaps and unknown subjects
raise). The loader writes `metadata['split_indices']`, and `TrajectoryDataset.split()`
uses those indices verbatim (ignoring random `seed`/`fractions`) whenever they exist —
so `nssc.experiment.prepare_data` and the CLI honour the subject split unchanged. Only
`by: subject` is supported for EEG; a random segment-level split would leak subject
identity between train and test.

### Configs

* `configs/datasets/eegbci.yaml` — subjects 1–8, runs [3, 7, 11], 64 ch, 64 Hz, 8 s
  segments (T=512): **x = (360, 512, 64)**, 45 segments/subject; split train S1–5 (225) /
  val S6 (45) / test S7–8 (90). First build (24 EDF downloads + filtering) took ~4.5 min
  wall on a laptop; cached rebuild ~2 s.
* `configs/datasets/eegbci_tiny.yaml` — subjects 1–3, run [3], first 8 channels, 32 Hz,
  band 1–14 Hz, 4 s segments (T=128): **x = (92, 128, 8)**, split train S1 (31) / val S2
  (30) / test S3 (31). Smoke/CI only: one subject per split has no statistical power.
* `configs/experiments/eegbci_smoke.yaml` — 2-epoch MLP-AE + residual-MLP run on tiny
  (tag `eeg_smoke`); it runs on `device: cpu` because the stability analysis uses
  `torch.linalg.eigvals` in float64, which MPS does not support.

Tests: `tests/unit/test_real_data.py` (offline: split logic, config resolution/hash,
segmentation, dispatch; the download test is opt-in via `NSSC_NETWORK_TESTS=1`).

### Caveats (read before making claims)

* **No claims about biological meaning of latents.** A compiled state-space model that
  predicts EEG well says nothing about neural mechanisms; latents are statistical
  summaries of filtered scalp potentials, not "brain states".
* Cross-subject generalisation is hard by construction (electrode placement, impedance,
  anatomy). Expect much worse held-out-subject numbers than within-subject; this is a
  feature of the protocol, not a bug of the model.
* Some PhysioNet subjects have known annotation/sampling irregularities (S088, S092,
  S100, S104 are commonly excluded in the literature); the loader does not exclude them
  automatically — choose subjects explicitly.
* Segments cut across task/rest boundaries; we do not align to events. Fine for
  unsupervised dynamics modelling, wrong for BCI classification claims.
* Filtering and resampling are done per run before segmentation, so segment edges carry
  no filter transients, but the first/last ~1 s of each *run* may.
* Non-stationarity across runs and subjects means the "dynamics" are at best locally
  stationary; treat rollout metrics beyond a few seconds with suspicion.
