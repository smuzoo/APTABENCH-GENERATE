"""Utility helpers for paths and project config."""

from pathlib import Path


def get_project_root() -> Path:
    """Return project root (parent of src)."""
    return Path(__file__).resolve().parent.parent


def get_data_path(filename: str = "AptaBench_dataset_v2.csv") -> Path:
    """Return path to data file."""
    return get_project_root() / "data" / filename


def get_models_path(filename: str = "lgbm_model.txt") -> Path:
    """Return path to saved model file."""
    return get_project_root() / "models" / filename
