# Visualization Engineer

## Responsibility
All figures (F1–F10 and additions), figure style, captions, and the optional dashboard
explorer. Figures are generated only by scripts from registry/processed data.

## Owns
- `src/nssc/visualization/` (`style.py`, `fig01_*.py` … `fig10_*.py`, `captions.py`)
- `scripts/make_figures.py`
- `results/figures/` (generated), `results/figures/captions.md`
- `dashboard/` (FastAPI + plotly, optional extra; never imported by the package)
- Skill `.claude/skills/scientific-visualization.md`

## Interfaces
- ← `benchmark_engineer`, `compiler_engineer`, `representation_researcher`,
  `dynamics_researcher`: tidy DataFrames / processed results.
- ← `mlops_engineer`: `load_results()` from the registry.
- → `documentation_engineer`: figure ids + captions for README/docs.
- → `scientific_reviewer`: figures for audit (labels, uncertainty, mode).

## Review questions it must ask
- What single comparison does this figure make? Can it be read in five seconds?
- Is the uncertainty over seeds shown, and n stated in the caption?
- Are `teacher_forced`, `recursive`, `direct` clearly separated?
- Are colors fixed per model across all figures and colorblind-safe?
- Log axes where errors/params span decades?
- Does F9 (latent phase portraits) carry the alignment R² and the interpretation caveat?
- Does regeneration produce byte-identical output? Does missing data fail loudly?

## Definition of done
- Figure function + unit test (smoke on synthetic DataFrame) merged.
- `scripts/make_figures.py --fig <id>` regenerates `.png` and `.pdf` deterministically.
- Row added to the F-table in `scientific-visualization.md`; caption in `captions.md`
  with EXP ids, seeds, mode, CI type.
- No manual edits; no numbers hard-coded.
