"""Task-Aware MIL framework for multi-sequence cardiac MRI slice selection."""

from .dataset import MILCardiacDataset, collate_fn, normalize_view_name
from .losses import MILLossWithSparsity
from .metrics import MILMetricAccumulator, format_metrics_line
from .model import (
    PositionalEncoding,
    SharedEncoder,
    TaskAwareMILModel,
    TaskSpecificClassifier,
    TransformerAttentionHead,
)
from .trainer import MILTrainer

__all__ = [
    "MILCardiacDataset",
    "collate_fn",
    "normalize_view_name",
    "MILLossWithSparsity",
    "MILMetricAccumulator",
    "format_metrics_line",
    "PositionalEncoding",
    "SharedEncoder",
    "TaskAwareMILModel",
    "TaskSpecificClassifier",
    "TransformerAttentionHead",
    "MILTrainer",
]
