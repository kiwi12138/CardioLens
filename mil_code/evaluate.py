"""
Evaluate a trained MIL checkpoint (Section E.3).

Reports per-task, per-sequence, and overall Accuracy, Precision, Recall, F1,
and Exact-match. Multi-label accuracy is the per-option agreement rate; P/R/F1
are micro-averaged across all option-level decisions.

Usage:
    python evaluate.py \\
        --checkpoint  ./checkpoints/best_model_acc.pth \\
        --cache-dir   /path/to/preprocessed_cache \\
        --label-json  /path/to/labels.json \\
        --output      ./val_metrics.json
"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import MILCardiacDataset, collate_fn
from src.metrics import MILMetricAccumulator, format_metrics_line
from src.model import TaskAwareMILModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained MIL checkpoint.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--label-json", required=True)
    p.add_argument("--output", required=True,
                   help="Path to write the metrics JSON.")

    p.add_argument("--split", default="val", choices=["train", "val"])
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-slices", type=int, default=512)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model = TaskAwareMILModel(
        all_task_configs=ckpt["all_task_configs"],
        backbone="resnet18",
        pretrained=False,
        target_k=ckpt["target_k"],
        num_heads=ckpt["num_heads"],
        use_self_attention=ckpt["use_self_attention"],
        use_positional_encoding=ckpt["use_positional_encoding"],
        learnable_pe=ckpt.get("learnable_pe", True),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(args.device).eval()
    sequence_tasks = ckpt["sequence_tasks"]
    print(f"Loaded checkpoint: {args.checkpoint}")

    dataset = MILCardiacDataset(
        cache_dir=args.cache_dir,
        label_json=args.label_json,
        split=args.split,
        val_ratio=args.val_ratio,
        random_seed=args.seed,
        use_augmentation=False,
        max_slices=args.max_slices,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    accumulator = MILMetricAccumulator(model.all_task_configs, sequence_tasks)

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating ({args.split})"):
            for sequence_type in batch["available_sequences"]:
                if sequence_type not in batch["images"]:
                    continue
                images = batch["images"][sequence_type].to(args.device)
                labels = batch["labels"][sequence_type]
                masks = batch["masks"][sequence_type]
                if all(masks[t] == 0 for t in masks):
                    continue

                seq_cfg = sequence_tasks[sequence_type]
                preds, _ = model.forward_sequence(images, sequence_type, seq_cfg)
                for task_name, logits in preds.items():
                    if masks.get(task_name, 0) == 0:
                        continue
                    accumulator.update(task_name, logits, labels[task_name])

    summary = accumulator.summary()

    print()
    print(format_metrics_line("OVERALL", summary["overall"]))
    for seq in sequence_tasks.keys():
        if summary[seq]["count"] > 0:
            print(format_metrics_line(seq, summary[seq]))
    print()
    print("Per-task breakdown:")
    for key, m in summary.items():
        if key.startswith("task::"):
            print(format_metrics_line(key.replace("task::", ""), m))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nMetrics written to {args.output}")


if __name__ == "__main__":
    main()
