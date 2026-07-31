"""Repo root on sys.path, so `kernels.*` resolves when these run as scripts from anywhere."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
