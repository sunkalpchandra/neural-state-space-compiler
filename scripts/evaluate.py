#!/usr/bin/env python3
"""Thin wrapper: `nssc evaluate`. See src/nssc/cli/main.py."""
import sys

from nssc.cli.main import app

if __name__ == "__main__":
    sys.argv.insert(1, "evaluate")
    app()
