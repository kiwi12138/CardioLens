"""
Entry point for training the Task-Aware MIL model.

Usage:
    python train.py \
        --cache-dir /path/to/preprocessed_cache \
        --label-json /path/to/labels.json \
        --checkpoint-dir ./checkpoints
"""
import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src.dataset import MILCardiacDataset, SEQUENCES, collate_fn
from src.losses import MILLossWithSparsity
from src.model import TaskAwareMILModel
from src.trainer import MILTrainer


def setup_logger(log_dir: str, experiment_name: str) -> logging.Logger:
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)
    log_file = log_dir_path / f"training_{experiment_name}.log"

    logger = logging.getLogger("MILTrainer")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt_full = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fmt_console = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt_full)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt_console)
    logger.addHandler(sh)

    logger.info(f"Log file: {log_file}")
    return logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Task-Aware MIL model.")
    p.add_argument("--cache-dir", required=True,
                   help="Path to preprocessed cache directory.")
    p.add_argument("--label-json", required=True,
                   help="Path to label / task config JSON.")
    p.add_argument("--checkpoint-dir", default="./checkpoints",
                   help="Where to write checkpoints, history, and logs.")

    # Optimization.
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-epochs", type=int, default=50)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--accumulation-steps", type=int, default=64)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)

    # Model architecture (paper Table 6 defaults).
    p.add_argument("--target-k", type=int, default=8)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--no-self-attention", action="store_true",
                   help="Disable per-task self-attention (ablation).")
    p.add_argument("--no-positional-encoding", action="store_true",
                   help="Disable positional encoding (ablation).")
    p.add_argument("--fixed-pe", action="store_true",
                   help="Use sinusoidal (fixed) instead of learnable PE.")

    # Loss weights.
    p.add_argument("--l1-weight", type=float, default=1.0)
    p.add_argument("--binary-weight", type=float, default=1.0)

    # Data.
    p.add_argument("--use-augmentation", action="store_true")
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--max-slices", type=int, default=512)

    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    torch.manual_seed(args.seed)

    experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logger(args.checkpoint_dir, experiment_name)

    logger.info("=" * 60)
    logger.info("Multi-sequence MIL training")
    logger.info("=" * 60)

    config = vars(args).copy()
    logger.info("Configuration:")
    for k, v in config.items():
        logger.info(f"  {k}: {v}")

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.checkpoint_dir) / f"config_{experiment_name}.json", "w") as f:
        json.dump(config, f, indent=2)

    # Build task configs grouped by sequence.
    with open(args.label_json, "r", encoding="utf-8") as f:
        label_data = json.load(f)
    all_task_configs = label_data["task_configs"]

    sequence_tasks = {s: {} for s in SEQUENCES}
    for task_name, cfg in all_task_configs.items():
        sequence_tasks[cfg["sequence"]][task_name] = {
            "type": cfg["type"],
            "num_classes": cfg["num_classes"],
        }

    logger.info(f"Total tasks: {len(all_task_configs)}")
    for s in SEQUENCES:
        logger.info(f"  {s}: {len(sequence_tasks[s])} tasks")

    # Datasets and loaders.
    train_dataset = MILCardiacDataset(
        cache_dir=args.cache_dir,
        label_json=args.label_json,
        split="train",
        val_ratio=args.val_ratio,
        random_seed=args.seed,
        use_augmentation=args.use_augmentation,
        max_slices=args.max_slices,
    )
    val_dataset = MILCardiacDataset(
        cache_dir=args.cache_dir,
        label_json=args.label_json,
        split="val",
        val_ratio=args.val_ratio,
        random_seed=args.seed,
        use_augmentation=False,
        max_slices=args.max_slices,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers,
    )

    # Model.
    model = TaskAwareMILModel(
        all_task_configs=all_task_configs,
        backbone="resnet18",
        pretrained=True,
        target_k=args.target_k,
        num_heads=args.num_heads,
        use_self_attention=not args.no_self_attention,
        use_positional_encoding=not args.no_positional_encoding,
        learnable_pe=not args.fixed_pe,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters:     {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    criterion = MILLossWithSparsity(
        all_task_configs,
        target_k=args.target_k,
        l1_weight=args.l1_weight,
        binary_weight=args.binary_weight,
    )
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    trainer = MILTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        sequence_tasks=sequence_tasks,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        accumulation_steps=args.accumulation_steps,
        grad_clip_norm=args.grad_clip_norm,
        logger=logger,
    )
    trainer.train(args.num_epochs)


if __name__ == "__main__":
    main()
