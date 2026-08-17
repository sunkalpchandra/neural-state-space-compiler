#!/usr/bin/env python3
"""Thin wrapper: `nssc train`. See src/nssc/cli/main.py."""
import sys

from nssc.cli.main import app

if __name__ == "__main__":
    sys.argv.insert(1, "train")
    app()
