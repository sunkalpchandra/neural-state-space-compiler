# Changelog

## 0.1.0 (in progress, 2026-08-17)
- Package bootstrap: registries, config system, experiment registry, CLI.
- 11 synthetic dynamical systems + Tier-2 observation maps; EEGBCI real-data loader.
- Encoders: PCA, linear, MLP, TCN, GRU/LSTM, diagonal SSM, multi-scale slow/fast.
- Dynamics: linear, affine, MLP, residual MLP, Koopman, neural ODE, SSM, multi-scale, Gaussian.
- Trainer with rollout curriculum, stability penalty, closed-form PCA/DMD init.
- Evaluator (recon / teacher-forced / recursive), stability analysis, uncertainty calibration,
  OOD evaluation, failure categorisation.
- StateSpaceCompiler: profiler → candidates → resumable staged search → multi-objective
  scorer (with error floor) → report; registry-level run reuse for ablations.
- Sequence-model baselines (persistence, GRU, LSTM, TCN, Transformer, SSM) + suite runner.
- Visualization package (script-generated figures), FastAPI/Plotly explorer dashboard.
