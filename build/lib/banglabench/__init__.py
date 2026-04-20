"""BanglaBayanno benchmark package."""

from .data import load_dataset
from .evaluator import evaluate
from .metrics import compute_metrics

__all__ = ["load_dataset", "evaluate", "compute_metrics"]
