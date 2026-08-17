"""Centralised metrics. All functions accept torch tensors or numpy arrays.

Shapes: predictions/targets are (B, T, D) unless stated. Returned values are
python floats or dicts of floats so they serialise directly into the registry.
"""

from nssc.metrics.calibration import (  # noqa: F401
    coverage,
    expected_calibration_error_regression,
    gaussian_nll,
)
from nssc.metrics.complexity import (  # noqa: F401
    count_parameters,
    estimate_flops_per_step,
    measure_inference_latency,
    peak_memory_mb,
)
from nssc.metrics.prediction import (  # noqa: F401
    horizon_curve,
    mse,
    nrmse,
    r2_score,
    rmse,
    rollout_divergence_time,
    rollout_errors,
)
