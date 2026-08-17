"""Every public subpackage must import in isolation (guards against import cycles)."""

import importlib
import subprocess
import sys

import pytest

MODULES = ["nssc.data", "nssc.representations", "nssc.dynamics", "nssc.models", "nssc.training",
           "nssc.evaluation", "nssc.metrics", "nssc.stability", "nssc.uncertainty", "nssc.search",
           "nssc.search.runner", "nssc.compiler", "nssc.baselines", "nssc.experiment", "nssc.cli.main"]


@pytest.mark.parametrize("mod", MODULES)
def test_import_in_fresh_interpreter(mod):
    r = subprocess.run([sys.executable, "-c", f"import {mod}"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-800:]


def test_import_all_in_process():
    for m in MODULES:
        importlib.import_module(m)
