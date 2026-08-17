"""Global matplotlib style for all nssc figures (headless, colorblind-safe, print-ready).

Import this module *before* ``matplotlib.pyplot`` anywhere in the visualization
package: it pins the Agg backend so figure generation works without a display.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

# Okabe–Ito colorblind-safe palette (Wong 2011).
OKABE_ITO: dict[str, str] = {
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#000000",
}
PALETTE: list[str] = [OKABE_ITO[k] for k in ("blue", "orange", "green", "vermillion", "purple",
                                             "sky", "yellow", "black")]

COLORS: dict[str, str] = {
    "true": "#222222",
    "pred": OKABE_ITO["vermillion"],
    "context": "#B0B0B0",
    "envelope": OKABE_ITO["sky"],
    "selected": OKABE_ITO["orange"],
    "front": OKABE_ITO["blue"],
    "grid": "#DDDDDD",
    "unit_circle": "#666666",
    **OKABE_ITO,
}

# Model *family* → fixed color across all figures.  Dynamics families take precedence
# for combined names like ``mlp+residual_mlp@d4`` (the dynamics is what differs on most
# comparison plots); baselines get their own set.
FAMILY_COLORS: dict[str, str] = {
    # dynamics families
    "linear": OKABE_ITO["blue"],
    "affine": OKABE_ITO["sky"],
    "mlp": OKABE_ITO["orange"],
    "residual_mlp": OKABE_ITO["vermillion"],
    "koopman": OKABE_ITO["green"],
    "neural_ode": OKABE_ITO["purple"],
    "ssm": OKABE_ITO["yellow"],
    "gaussian": "#8C564B",
    "multiscale": "#17BECF",
    # encoder families (used when a name is a bare encoder)
    "pca": OKABE_ITO["blue"],
    "tcn": OKABE_ITO["green"],
    "gru": OKABE_ITO["purple"],
    "linear_ae": OKABE_ITO["sky"],
    # baselines
    "baseline": "#7F7F7F",
    "baseline:gru": "#7F7F7F",
    "baseline:lstm": "#5C5C5C",
    "baseline:tcn": "#A6A6A6",
    "baseline:transformer": "#3D3D3D",
    "baseline:ssm": "#999999",
    "baseline:persistence": "#C7C7C7",
    "baseline:mean": "#C7C7C7",
    "persistence": "#C7C7C7",
    "compiled": OKABE_ITO["orange"],
}
MODEL_COLORS = FAMILY_COLORS  # alias used by the scientific-visualization skill

RC: dict[str, Any] = {
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#DDDDDD",
    "grid.linewidth": 0.5,
    "grid.alpha": 0.8,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.4,
    "lines.markersize": 4,
    "axes.prop_cycle": plt.cycler(color=PALETTE),
    "figure.autolayout": False,
    "figure.constrained_layout.use": True,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "image.cmap": "viridis",
}

SINGLE_COL = 3.4  # inches
DOUBLE_COL = 7.0


class _StyleContext:
    """Applies the nssc rc globally on creation; usable as ``with use_style(): ...``
    to restore the previous rc on exit."""

    def __init__(self) -> None:
        self._prev = dict(matplotlib.rcParams)
        matplotlib.rcParams.update(RC)

    def __enter__(self) -> _StyleContext:
        return self

    def __exit__(self, *exc: Any) -> None:
        matplotlib.rcParams.update(self._prev)


def use_style() -> _StyleContext:
    """Apply the project style. Works both as a plain call and as a context manager."""
    return _StyleContext()


def family_of(name: str) -> str:
    """Model family key for ``name``: ``baseline:<k>`` → baseline key; ``enc+dyn@dN`` → dyn."""
    n = str(name)
    if n.startswith("baseline"):
        key = n.split("/")[0]
        return key if key in FAMILY_COLORS else "baseline"
    if "+" in n:
        dyn = n.split("+", 1)[1].split("@")[0]
        return dyn
    return n.split("@")[0]


def model_color(name: str) -> str:
    """Deterministic color for a model/candidate name (fixed per family; hashed fallback)."""
    fam = family_of(name)
    if fam in FAMILY_COLORS:
        return FAMILY_COLORS[fam]
    h = int(hashlib.sha1(fam.encode()).hexdigest(), 16)  # noqa: S324 - not security
    return PALETTE[h % len(PALETTE)]


def save(fig: plt.Figure, path: str | Path, formats: Iterable[str] = ("png", "pdf"),
         close: bool = True, dpi: int = 300) -> list[Path]:
    """Save ``fig`` as ``<path stem>.<fmt>`` for every format. Returns the written paths.

    A suffix on ``path`` is ignored (stem is used) so ``save(fig, "a.png")`` and
    ``save(fig, "a")`` write the same files. PDFs get fixed metadata for byte-stable output.
    """
    path = Path(path)
    stem = path.with_suffix("") if path.suffix.lower() in (".png", ".pdf", ".svg") else path
    stem.parent.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    for fmt in formats:
        p = stem.with_suffix(f".{fmt}")
        kw: dict[str, Any] = {"dpi": dpi, "bbox_inches": "tight"}
        if fmt == "pdf":
            kw["metadata"] = {"CreationDate": None, "Producer": "nssc", "Creator": "nssc"}
        fig.savefig(p, **kw)
        out.append(p)
    if close:
        plt.close(fig)
    return out


use_style()
