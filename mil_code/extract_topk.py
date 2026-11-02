"""
Importance-based slice selection for image understanding and report generation.

The MIL model produces a separate attention head per task, so a single forward
pass over a (patient, sequence) bag yields per-task importance weights. From
those weights we record:

  * per_task[pid][sequence][task]
        Top-K indices using that task's own attention head. This is the
        selection used for image-understanding evaluation, where each
        question is tied to a single (sequence, subtask) pair (paper
        Section D, "Image understanding (baseline case)").

  * per_sequence_task_averaged[pid][sequence]
        Top-K indices using the task-averaged importance score
        w_bar_i = (1/|T_s|) * sum_j w_{j,i}. This is the selection used for
        report generation and (after concatenation across sequences) for
        disease diagnosis.

Indices are bag-relative, i.e., they refer to the slices in the bag returned
by the dataset (the union of task-relevant views, optionally subsampled to
max_slices). The dataset's view-filtering rules are documented in
src/dataset.py.

Usage:
    python extract_topk.py \\
        --checkpoint  ./checkpoints/best_model_acc.pth \\
        --cache-dir   /path/to/preprocessed_cache \\
        --label-json  /path/to/labels.json \\
        --output      ./topk_indices.json \\
        --target-k    8
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import MILCardiacDataset, collate_fn
from src.model import TaskAwareMILModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-task and task-averaged top-K extraction.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--label-json", required=True)
    p.add_argument("--output", required=True,
                   help="Path to write the top-K index JSON.")

    p.add_argument("--split", default="val", choices=["train", "val"])
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-slices", type=int, default=512)

    p.add_argument("--target-k", type=int, default=None,
                   help="If unset, use the value stored in the checkpoint.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_model(checkpoint_path: str, device: str) -> Tuple[TaskAwareMILModel, dict]:
    ckpt = torch.load(checkpoint_path, map_location=device)
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
    model.to(device).eval()
    return model, ckpt


def topk_sorted(weights: torch.Tensor, k: int) -> List[int]:
    """Return the indices of the k largest entries in ascending order."""
    k = min(k, weights.shape[0])
    idx = torch.topk(weights, k=k, largest=True).indices
    idx_sorted, _ = torch.sort(idx)
    return idx_sorted.cpu().tolist()


def main() -> None:
    args = parse_args()

    model, ckpt = load_model(args.checkpoint, args.device)
    sequence_tasks = ckpt["sequence_tasks"]
    target_k = args.target_k if args.target_k is not None else ckpt["target_k"]
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Target K = {target_k}")

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

    # patient_id -> sequence -> task_name -> [bag-relative indices]
    per_task: Dict[str, Dict[str, Dict[str, List[int]]]] = {}
    # patient_id -> sequence -> [bag-relative indices]
    per_seq_avg: Dict[str, Dict[str, List[int]]] = {}

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting top-K"):
            pid = batch["patient_id"]
            per_task.setdefault(pid, {})
            per_seq_avg.setdefault(pid, {})

            for sequence_type in batch["available_sequences"]:
                if sequence_type not in batch["images"]:
                    continue
                images = batch["images"][sequence_type].to(args.device)
                seq_cfg = sequence_tasks.get(sequence_type, {})
                if not seq_cfg:
                    continue

                # Single forward pass; per-task heads expose their own weights.
                _, attn_info = model.forward_sequence(images, sequence_type, seq_cfg)

                # Per-task selection (image understanding).
                task_block: Dict[str, List[int]] = {}
                weight_stack: List[torch.Tensor] = []
                for task_name, info in attn_info.items():
                    w = info["weights"]                       # [N]
                    task_block[task_name] = topk_sorted(w, target_k)
                    weight_stack.append(w)

                if task_block:
                    per_task[pid][sequence_type] = task_block

                # Task-averaged selection (report generation; Algorithm 2).
                if weight_stack:
                    mean_w = torch.stack(weight_stack, dim=0).mean(dim=0)  # [N]
                    per_seq_avg[pid][sequence_type] = topk_sorted(mean_w, target_k)

    output = {
        "target_k": target_k,
        "per_task": per_task,
        "per_sequence_task_averaged": per_seq_avg,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    n_pat = len(per_task)
    n_task_sel = sum(
        len(seq_block) for seq_map in per_task.values() for seq_block in seq_map.values()
    )
    n_seq_sel = sum(len(seq_map) for seq_map in per_seq_avg.values())
    print(
        f"Wrote selections for {n_pat} patients to {args.output}\n"
        f"  per-task entries:              {n_task_sel}\n"
        f"  task-averaged sequence entries: {n_seq_sel}"
    )


if __name__ == "__main__":
    main()
