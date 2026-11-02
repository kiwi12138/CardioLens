"""
Training loop for the Task-Aware MIL model.

"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .losses import MILLossWithSparsity
from .metrics import MILMetricAccumulator, format_metrics_line
from .model import TaskAwareMILModel


class MILTrainer:
    def __init__(
        self,
        model: TaskAwareMILModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: MILLossWithSparsity,
        optimizer: optim.Optimizer,
        sequence_tasks: Dict,
        device: str = "cuda",
        checkpoint_dir: str = "./checkpoints",
        accumulation_steps: int = 64,
        grad_clip_norm: float = 1.0,
        logger: logging.Logger = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.sequence_tasks = sequence_tasks
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.accumulation_steps = accumulation_steps
        self.grad_clip_norm = grad_clip_norm
        self.logger = logger or logging.getLogger("MILTrainer")

        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0
        self.train_history = []
        self.val_history = []

    # ------------------------------------------------------------------ #
    # Train / validate loops
    # ------------------------------------------------------------------ #
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()

        epoch_losses = {t: 0.0 for t in self.model.all_task_configs.keys()}
        epoch_losses.update({"total": 0.0, "task": 0.0, "l1": 0.0, "binary": 0.0})
        seq_counts = {s: 0 for s in self.sequence_tasks.keys()}
        n_steps = 0

        weight_stats = {"sum": 0.0, "max": 0.0, "min": 1.0, "n": 0}

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            available = batch["available_sequences"]
            if not available:
                continue
            sequence_type = np.random.choice(available)
            if sequence_type not in batch["images"]:
                continue

            images = batch["images"][sequence_type].to(self.device)
            labels = {k: v.to(self.device) for k, v in batch["labels"][sequence_type].items()}
            masks = batch["masks"][sequence_type]
            if all(masks[t] == 0 for t in masks):
                continue

            seq_task_cfg = self.sequence_tasks[sequence_type]
            preds, attn_info = self.model.forward_sequence(
                images, sequence_type, seq_task_cfg
            )
            loss, loss_dict = self.criterion(preds, labels, masks, attn_info)

            if not loss.requires_grad:
                continue

            (loss / self.accumulation_steps).backward()
            if (batch_idx + 1) % self.accumulation_steps == 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()

            epoch_losses["total"] += loss.item()
            for k, v in loss_dict.items():
                if k in epoch_losses:
                    epoch_losses[k] += v
            seq_counts[sequence_type] += 1
            n_steps += 1

            for info in attn_info.values():
                w = info["weights"].detach()
                weight_stats["sum"] += w.sum().item()
                weight_stats["max"] = max(weight_stats["max"], w.max().item())
                weight_stats["min"] = min(weight_stats["min"], w.min().item())
                weight_stats["n"] += 1

            del images, preds, loss
            if batch_idx % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

            if n_steps > 0:
                pbar.set_postfix({
                    "loss": epoch_losses["total"] / n_steps,
                    "task": epoch_losses["task"] / n_steps,
                    "l1": epoch_losses["l1"] / n_steps,
                    "bin": epoch_losses["binary"] / n_steps,
                })

        # Flush the final partial accumulation step.
        if n_steps % self.accumulation_steps != 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()

        if weight_stats["n"] > 0:
            self.logger.info(
                f"[Train] Weight stats - "
                f"avg sum: {weight_stats['sum'] / weight_stats['n']:.3f}, "
                f"max: {weight_stats['max']:.3f}, "
                f"min: {weight_stats['min']:.3f}"
            )
        self.logger.info(
            "[Train] Sequence distribution - "
            + ", ".join(f"{s}: {n}" for s, n in seq_counts.items())
        )

        return {k: v / max(n_steps, 1) for k, v in epoch_losses.items()}

    def validate(self) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
        self.model.eval()

        epoch_losses = {t: 0.0 for t in self.model.all_task_configs.keys()}
        epoch_losses.update({"total": 0.0, "task": 0.0, "l1": 0.0, "binary": 0.0})
        seq_counts = {s: 0 for s in self.sequence_tasks.keys()}
        n_steps = 0

        accumulator = MILMetricAccumulator(
            self.model.all_task_configs, self.sequence_tasks
        )

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                available = batch["available_sequences"]
                for sequence_type in available:
                    if sequence_type not in batch["images"]:
                        continue
                    images = batch["images"][sequence_type].to(self.device)
                    labels = {k: v.to(self.device) for k, v in batch["labels"][sequence_type].items()}
                    masks = batch["masks"][sequence_type]
                    if all(masks[t] == 0 for t in masks):
                        continue

                    seq_task_cfg = self.sequence_tasks[sequence_type]
                    preds, attn_info = self.model.forward_sequence(
                        images, sequence_type, seq_task_cfg
                    )
                    loss, loss_dict = self.criterion(preds, labels, masks, attn_info)

                    epoch_losses["total"] += loss.item()
                    for k, v in loss_dict.items():
                        if k in epoch_losses:
                            epoch_losses[k] += v
                    seq_counts[sequence_type] += 1
                    n_steps += 1

                    for task_name, logits in preds.items():
                        if masks.get(task_name, 0) == 0:
                            continue
                        accumulator.update(task_name, logits, labels[task_name])

                    del images, preds, loss

        avg_losses = {k: v / max(n_steps, 1) for k, v in epoch_losses.items()}
        metrics = accumulator.summary()

        # Logging.
        self.logger.info(
            "[Val] Sequence distribution - "
            + ", ".join(f"{s}: {n}" for s, n in seq_counts.items())
        )
        self.logger.info(format_metrics_line("OVERALL", metrics["overall"]))
        for seq in self.sequence_tasks.keys():
            if metrics[seq]["count"] > 0:
                self.logger.info(format_metrics_line(seq, metrics[seq]))

        return avg_losses, metrics

    # ------------------------------------------------------------------ #
    # Top-level training entry point
    # ------------------------------------------------------------------ #
    def train(self, num_epochs: int) -> None:
        self.logger.info("=" * 60)
        self.logger.info(f"Starting training for {num_epochs} epochs")
        self.logger.info("=" * 60)

        self.logger.info("Initial validation before training:")
        val_losses, val_metrics = self.validate()

        for epoch in range(num_epochs):
            t0 = datetime.now()
            self.logger.info("=" * 60)
            self.logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            self.logger.info("=" * 60)

            train_losses = self.train_epoch(epoch + 1)
            self.train_history.append(train_losses)
            self.logger.info(
                f"[Train] total: {train_losses['total']:.4f} | "
                f"task: {train_losses['task']:.4f} | "
                f"l1: {train_losses['l1']:.4f} | "
                f"bin: {train_losses['binary']:.4f}"
            )

            val_losses, val_metrics = self.validate()
            self.val_history.append({"losses": val_losses, "metrics": val_metrics})

            overall = val_metrics["overall"]
            self.logger.info(
                f"[Val] total: {val_losses['total']:.4f} | "
                f"acc: {overall['accuracy']:.4f} | "
                f"f1: {overall['f1']:.4f} | "
                f"em: {overall['exact_match']:.4f}"
            )

            if val_losses["total"] < self.best_val_loss:
                self.best_val_loss = val_losses["total"]
                self.save_checkpoint(epoch, "best_model_loss.pth")
                self.logger.info(f"  >> Best model (loss): {self.best_val_loss:.4f}")
            if overall["accuracy"] > self.best_val_acc:
                self.best_val_acc = overall["accuracy"]
                self.save_checkpoint(epoch, "best_model_acc.pth")
                self.logger.info(f"  >> Best model (acc): {self.best_val_acc:.4f}")
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, f"checkpoint_epoch_{epoch + 1}.pth")

            self.logger.info(f"Epoch time: {datetime.now() - t0}")

        self.logger.info("=" * 60)
        self.logger.info("Training complete.")
        self.logger.info(f"Best val loss: {self.best_val_loss:.4f}")
        self.logger.info(f"Best val acc:  {self.best_val_acc:.4f}")
        self.logger.info("=" * 60)
        self.save_training_history()

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save_checkpoint(self, epoch: int, filename: str) -> None:
        any_head = next(iter(self.model.attention_heads.values()))
        learnable_pe = (
            isinstance(self.model.pos_encoding.pe, nn.Parameter)
            if self.model.use_positional_encoding
            else False
        )
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "best_val_acc": self.best_val_acc,
            "all_task_configs": self.model.all_task_configs,
            "sequence_tasks": self.sequence_tasks,
            "target_k": self.model.target_k,
            "num_heads": any_head.num_heads,
            "use_self_attention": any_head.use_self_attention,
            "use_positional_encoding": self.model.use_positional_encoding,
            "learnable_pe": learnable_pe,
            "l1_weight": self.criterion.l1_weight,
            "binary_weight": self.criterion.binary_weight,
        }
        torch.save(checkpoint, self.checkpoint_dir / filename)

    def save_training_history(self) -> None:
        history = {
            "train": self.train_history,
            "val": self.val_history,
            "best_val_loss": self.best_val_loss,
            "best_val_acc": self.best_val_acc,
        }
        with open(self.checkpoint_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2, default=lambda o: float(o))
