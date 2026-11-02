"""
Evaluation metrics.

Per-instance accuracy:
    - Single-label task: 1 if argmax(pred) == y else 0 (exact match).
    - Multi-label task with option set U:
        Acc = |{o in U : 1[o in y_hat] == 1[o in y]}| / |U|
      i.e., fraction of options that are correctly included or excluded.

Precision / Recall / F1 are micro-averaged across all instances by accumulating
TP / FP / FN across instances and combining.

Exact-match (Table 7 of the paper) is reported separately:
    - Single-label: equal to accuracy.
    - Multi-label: 1 if the predicted positive set equals the ground-truth set.
"""
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch


def _multilabel_predict(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    Threshold-based decoding for multi-label tasks.
    Falls back to argmax when the threshold yields the empty set, so that
    every instance produces at least one positive label.
    """
    probs = torch.sigmoid(logits)
    pred = (probs > threshold).float()
    if pred.sum() == 0:
        pred = torch.zeros_like(pred)
        pred[torch.argmax(probs)] = 1.0
    return pred


def _singlelabel_predict(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    """One-hot encoding of the argmax class. Lets us treat single- and multi-label
    tasks with a unified per-option counting scheme."""
    pred_idx = int(torch.argmax(logits).item())
    pred = torch.zeros(num_classes)
    pred[pred_idx] = 1.0
    return pred


class MILMetricAccumulator:
    """
    Accumulates per-option TP/FP/FN/TN counts across instances and provides
    micro-averaged Accuracy, Precision, Recall, F1, and Exact-match.

    Counts can be queried at three granularities:
        - per task,
        - per sequence (Cine / LGE / T2 / Perfusion),
        - overall.
    """

    def __init__(self, task_configs: Dict, sequence_tasks: Dict):
        self.task_configs = task_configs
        self.sequence_tasks = sequence_tasks

        # Per-option counters (for accuracy / precision / recall / F1).
        self._tp = defaultdict(int)
        self._fp = defaultdict(int)
        self._fn = defaultdict(int)
        self._tn = defaultdict(int)
        # Per-instance counters (for exact-match accuracy).
        self._exact_correct = defaultdict(int)
        self._exact_total = defaultdict(int)

    def update(
        self,
        task_name: str,
        pred_logits: torch.Tensor,
        true_label: torch.Tensor,
    ) -> None:
        cfg = self.task_configs[task_name]
        num_classes = cfg["num_classes"]

        if cfg["type"] == "single":
            pred_vec = _singlelabel_predict(pred_logits.detach().cpu(), num_classes)
            true_idx = int(true_label.item())
            true_vec = torch.zeros(num_classes)
            true_vec[true_idx] = 1.0
        elif cfg["type"] == "multi":
            pred_vec = _multilabel_predict(pred_logits.detach().cpu())
            true_vec = true_label.detach().cpu().float()
        else:
            raise ValueError(f"Unknown task type: {cfg['type']}")

        pred_np = pred_vec.numpy().astype(bool)
        true_np = true_vec.numpy().astype(bool)

        tp = int(np.logical_and(pred_np, true_np).sum())
        fp = int(np.logical_and(pred_np, ~true_np).sum())
        fn = int(np.logical_and(~pred_np, true_np).sum())
        tn = int(np.logical_and(~pred_np, ~true_np).sum())

        self._tp[task_name] += tp
        self._fp[task_name] += fp
        self._fn[task_name] += fn
        self._tn[task_name] += tn

        self._exact_total[task_name] += 1
        if np.array_equal(pred_np, true_np):
            self._exact_correct[task_name] += 1

    @staticmethod
    def _scores_from_counts(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
        denom = tp + fp + fn + tn
        accuracy = (tp + tn) / denom if denom > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def task_metrics(self, task_name: str) -> Dict[str, float]:
        scores = self._scores_from_counts(
            self._tp[task_name],
            self._fp[task_name],
            self._fn[task_name],
            self._tn[task_name],
        )
        total = self._exact_total[task_name]
        scores["exact_match"] = (
            self._exact_correct[task_name] / total if total > 0 else 0.0
        )
        scores["count"] = total
        return scores

    def _aggregate(self, task_names: List[str]) -> Dict[str, float]:
        tp = sum(self._tp[t] for t in task_names)
        fp = sum(self._fp[t] for t in task_names)
        fn = sum(self._fn[t] for t in task_names)
        tn = sum(self._tn[t] for t in task_names)
        scores = self._scores_from_counts(tp, fp, fn, tn)

        ec = sum(self._exact_correct[t] for t in task_names)
        et = sum(self._exact_total[t] for t in task_names)
        scores["exact_match"] = ec / et if et > 0 else 0.0
        scores["count"] = et
        return scores

    def sequence_metrics(self, sequence_name: str) -> Dict[str, float]:
        names = [
            t
            for t in self.sequence_tasks.get(sequence_name, {}).keys()
            if self._exact_total[t] > 0
        ]
        return self._aggregate(names) if names else {
            "accuracy": 0.0, "precision": 0.0, "recall": 0.0,
            "f1": 0.0, "exact_match": 0.0, "count": 0,
        }

    def overall_metrics(self) -> Dict[str, float]:
        names = [t for t in self.task_configs.keys() if self._exact_total[t] > 0]
        return self._aggregate(names)

    def summary(self) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {"overall": self.overall_metrics()}
        for seq in self.sequence_tasks.keys():
            out[seq] = self.sequence_metrics(seq)
        for task in self.task_configs.keys():
            if self._exact_total[task] > 0:
                out[f"task::{task}"] = self.task_metrics(task)
        return out


def format_metrics_line(name: str, m: Dict[str, float]) -> str:
    return (
        f"{name:>10s} | "
        f"Acc: {m['accuracy']:.4f} | "
        f"P: {m['precision']:.4f} | "
        f"R: {m['recall']:.4f} | "
        f"F1: {m['f1']:.4f} | "
        f"EM: {m['exact_match']:.4f} | "
        f"N: {m['count']}"
    )
