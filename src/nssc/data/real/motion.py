"""Motion-capture (CMU MoCap) real-data source — NOT IMPLEMENTED (honest stub).

We deliberately ship no synthetic stand-in here: a "motion" dataset that is not real
motion would silently pollute Tier-3 real-world claims. Implementing this source needs:

* Data: CMU Graphics Lab Motion Capture Database (http://mocap.cs.cmu.edu, free for
  research; cite "The data used in this project was obtained from mocap.cs.cmu.edu.
  The database was created with funding from NSF EIA-0196217"). Files are ASF/AMC
  (skeleton + per-frame joint angles, 120 Hz) or the pre-converted BVH / .c3d releases.
* Parsing: an ASF/AMC (or BVH) reader producing per-frame joint-angle vectors
  ``(T, D)`` (D≈62 for the CMU skeleton) — e.g. via a small vendored parser; there is
  no maintained lightweight PyPI dependency we want at package-import time.
* Preprocessing: subject/trial selection by motion category (walk/run/jump ...),
  root-position/orientation removal, downsampling (120 → 30/60 Hz), fixed-length
  segmentation as in :mod:`nssc.data.real.eegbci`.
* Split protocol: subject-level (never mix a subject's trials across splits) with the
  same ``metadata['split_indices']`` mechanism used by the EEG loader.
* Config schema mirroring ``source: eegbci`` (``subjects``, ``trials``/``categories``,
  ``resample_hz``, ``segment_seconds``, ``split``, ``cache_dir``).

Until that exists, ``build_motion`` raises ``NotImplementedError``.
"""

from __future__ import annotations

from typing import Any

SOURCE = "motion_cmu_mocap"


def build_motion(cfg: dict[str, Any]):  # noqa: ARG001 - signature kept for the dispatcher
    """Placeholder: raises ``NotImplementedError`` (see module docstring for the plan)."""
    raise NotImplementedError(
        "source 'motion_cmu_mocap' is not implemented yet: needs a CMU MoCap ASF/AMC "
        "parser + subject-level split (see nssc/data/real/motion.py docstring). "
        "No synthetic placeholder is provided on purpose.")
