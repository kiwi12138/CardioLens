"""
Multi-task MIL loss with target-K sparsity regularization.

Implements the training objective from Section C.2 of the paper:

    L = L_task
        + (lambda_1 / |T_valid|) * sum_j |sum_i w_{j,i} - K|
        + (lambda_2 / |T_valid|) * sum_j sum_i w_{j,i} * (1 - w_{j,i})

where T_valid is the set of tasks with available labels in the current bag.
"""
from typing import Dict, Tuple

import torch
import torch.nn as nn


class MILLossWithSparsity(nn.Module):
    def __init__(
        self,
        task_configs: Dict,
        target_k: int = 8,
        l1_weight: float = 1.0,
        binary_weight: float = 1.0,
    ):
        super().__init__()
        self.task_configs = task_configs
        self.target_k = target_k
        self.l1_weight = l1_weight
        self.binary_weight = binary_weight

        self.ce_loss = nn.CrossEntropyLoss(reduction="none")
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")

        print("MIL loss with target-K sparsity:")
        print(f"  Target K: {target_k}")
        print(f"  L1 weight: {l1_weight}")
        print(f"  Binary weight: {binary_weight}")

    @staticmethod
    def _l1(weights: torch.Tensor, target_k: int) -> torch.Tensor:
        # |sum(w) - K|
        return torch.abs(weights.sum() - target_k)

    @staticmethod
    def _binary(weights: torch.Tensor) -> torch.Tensor:
        # sum(w * (1 - w)); minimized when each w is 0 or 1.
        return torch.sum(weights * (1.0 - weights))

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor],
        masks: Dict[str, int],
        attention_info: Dict[str, Dict],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if not predictions:
            return (
                torch.tensor(0.0, requires_grad=True),
                {"l1": 0.0, "binary": 0.0, "task": 0.0},
            )

        device = next(iter(predictions.values())).device
        total_task = torch.tensor(0.0, device=device)
        total_l1 = torch.tensor(0.0, device=device)
        total_bin = torch.tensor(0.0, device=device)

        loss_dict: Dict[str, float] = {}
        n_valid = 0

        for task_name, cfg in self.task_configs.items():
            if masks.get(task_name, 0) == 0 or task_name not in predictions:
                loss_dict[task_name] = 0.0
                continue

            pred = predictions[task_name]
            label = labels[task_name]

            if cfg["type"] == "single":
                pred_b = pred.unsqueeze(0)
                label_b = label.unsqueeze(0) if label.dim() == 0 else label
                task_loss = self.ce_loss(pred_b, label_b).mean()
            elif cfg["type"] == "multi":
                task_loss = self.bce_loss(pred, label).mean()
            else:
                raise ValueError(f"Unknown task type: {cfg['type']}")

            total_task = total_task + task_loss
            loss_dict[task_name] = task_loss.item()

            if task_name in attention_info:
                w = attention_info[task_name]["weights"]
                total_l1 = total_l1 + self._l1(w, self.target_k)
                total_bin = total_bin + self._binary(w)

            n_valid += 1

        if n_valid > 0:
            avg_l1 = total_l1 / n_valid
            avg_bin = total_bin / n_valid
            total = total_task + self.l1_weight * avg_l1 + self.binary_weight * avg_bin
        else:
            avg_l1 = torch.tensor(0.0, device=device)
            avg_bin = torch.tensor(0.0, device=device)
            total = torch.tensor(0.0, device=device, requires_grad=True)

        loss_dict["task"] = total_task.item()
        loss_dict["l1"] = avg_l1.item()
        loss_dict["binary"] = avg_bin.item()
        return total, loss_dict
