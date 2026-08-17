"""Script-generated figures (matplotlib, Agg backend pinned in ``style``)."""

from nssc.visualization.style import (  # isort: skip  (must run first: pins Agg backend)
    COLORS,
    FAMILY_COLORS,
    MODEL_COLORS,
    family_of,
    model_color,
    save,
    use_style,
)

# isort: split
from nssc.visualization.cli_hooks import visualize_experiment
from nssc.visualization.compiler_plots import (
    plot_compiler_decision,
    plot_family_comparison,
    plot_latent_dim_sweep,
    plot_stage_funnel,
)
from nssc.visualization.figures import (
    figures_for_compile,
    figures_for_experiment,
    figures_for_suite,
    plot_training_curves,
)
from nssc.visualization.latent import (
    align_latents,
    plot_latent_trajectories,
    plot_latent_vs_true,
    plot_phase_portrait,
)
from nssc.visualization.pareto import plot_pareto
from nssc.visualization.rollout import (
    plot_horizon_curves,
    plot_one_step_vs_long_horizon,
    plot_rollout_comparison,
)
from nssc.visualization.stability import (
    plot_eigenvalue_spectrum,
    plot_norm_growth,
    plot_spectral_radius_hist,
    plot_vector_field,
)

__all__ = [
    "COLORS", "FAMILY_COLORS", "MODEL_COLORS", "family_of", "model_color", "save", "use_style",
    "align_latents", "plot_latent_trajectories", "plot_latent_vs_true", "plot_phase_portrait",
    "plot_horizon_curves", "plot_one_step_vs_long_horizon", "plot_rollout_comparison",
    "plot_pareto",
    "plot_eigenvalue_spectrum", "plot_norm_growth", "plot_spectral_radius_hist", "plot_vector_field",
    "plot_compiler_decision", "plot_family_comparison", "plot_latent_dim_sweep", "plot_stage_funnel",
    "figures_for_compile", "figures_for_experiment", "figures_for_suite", "plot_training_curves",
    "visualize_experiment",
]
