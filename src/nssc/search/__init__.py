"""Candidate space, resumable search state and staged search over latent state-space models."""

from nssc.search.space import CandidateSpec, generate_candidates  # noqa: F401
from nssc.search.state import SearchState  # noqa: F401


def __getattr__(name):  # lazy: staged imports nssc.compiler which imports search
    if name == "StagedSearch":
        from nssc.search.staged import StagedSearch

        return StagedSearch
    raise AttributeError(name)
