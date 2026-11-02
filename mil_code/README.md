# Task-Aware MIL for Cardiac MRI Slice Selection

Reference implementation of the importance-based slice selection framework
described in Section 2.4 / Appendix C of the paper. The MIL model produces
per-slice, per-task importance weights from sequence-level supervision, and
its top-K outputs serve as inputs to downstream MLLM evaluation.

## Repository layout

```
.
├── src/
│   ├── __init__.py
│   ├── model.py        # Encoder + per-task transformer attention + classifier
│   ├── dataset.py      # Multi-sequence cardiac MRI dataset
│   ├── losses.py       # Multi-task loss + target-K sparsity regularization
│   ├── metrics.py      # Acc / Precision / Recall / F1 / EM
│   └── trainer.py      # Algorithm 1 training loop
├── train.py            # Entry point: train the MIL model
├── extract_topk.py     # Per-task and task-averaged top-K slice extraction
├── evaluate.py         # Stand-alone evaluation 
└── README.md
```

## Alignment with the paper

The implementation follows Section C of the paper. Key design choices:

| Component           | Paper section              | Notes                                          |
|---------------------|----------------------------|------------------------------------------------|
| Backbone            | C.1.2 / Table 6            | ResNet-18, ImageNet pretrained, 1-channel.     |
| Positional encoding | C.1.3                      | Learnable, max length 1000.                    |
| Self-attention      | C.1.4 stage 1              | 4 heads, dropout 0.1.                          |
| Task query          | C.1.4 stage 2              | One learnable query per task.                  |
| Slice scorer        | C.1.4 stage 3, eq. (7)     | MLP + sigmoid (not softmax).                   |
| L1 sparsity         | C.2.1 eq. (9)              | `\|sum(w) - K\|`.                              |
| Binary sparsity     | C.2.1 eq. (10)             | `sum(w * (1 - w))`.                            |
| Total objective     | C.2.2 eq. (11)             | `L_task + (lambda_1/|T_valid|) L_sum + ...`.   |
| Train / val split   | C.3.5                      | 80% / 20% at the patient level.                |
| Top-K extraction    | Algorithm 2                | Task-averaged weights + `topk`.                |


## Quick start

```bash
# 1. Train the MIL model.
python train.py \
    --cache-dir   /path/to/preprocessed_cache \
    --label-json  /path/to/labels.json \
    --checkpoint-dir ./checkpoints

# 2. Evaluate the trained checkpoint.
python evaluate.py \
    --checkpoint  ./checkpoints/best_model_acc.pth \
    --cache-dir   /path/to/preprocessed_cache \
    --label-json  /path/to/labels.json \
    --output      ./val_metrics.json

# 3. Extract top-K slices for downstream MLLM evaluation.
python extract_topk.py \
    --checkpoint  ./checkpoints/best_model_acc.pth \
    --cache-dir   /path/to/preprocessed_cache \
    --label-json  /path/to/labels.json \
    --output      ./topk_indices.json \
    --target-k    8
```

### Top-K output format

`extract_topk.py` writes both selections needed by the downstream evaluation
tiers (paper Section D):

```json
{
  "target_k": 8,
  "per_task": {
    "<patient_id>": {
      "<sequence>": {"<task_name>": [int, ...]}
    }
  },
  "per_sequence_task_averaged": {
    "<patient_id>": {"<sequence>": [int, ...]}
  }
}
```

Use `per_task` for **image understanding** evaluation, where each question
is tied to a single (sequence, subtask) pair. Use
`per_sequence_task_averaged` for **report generation** (one report per
sequence covers all of that sequence's subtasks); for **disease diagnosis**
the four sequences' task-averaged selections are concatenated to form the
4K-slice patient input. All indices are bag-relative (i.e., they refer to
the slices in the bag returned by the dataset after view filtering and
optional `max_slices` subsampling).

## Expected data layout

```
cache_dir/
    index.json                          # {pid: {"sequences": [...]}}
    <pid>/
        metadata.json                   # {seq: [{"view": "..."}]}
        cine.npy                        # [N, H, W] in [0, 1]
        LGE.npy
        T2.npy
        perfusion.npy
```

The label JSON should provide:

```json
{
  "task_configs": {
    "<task_name>": {
      "sequence":   "cine|LGE|T2|perfusion",
      "type":       "single|multi",
      "num_classes": 2,
      "choices":    ["...", "..."],
      "views":      ["sax", "4ch"]
    }
  },
  "labels": {
    "<patient_id>": {"<task_name>": ["<choice_text>", "..."]}
  },
  "patient_id_list": ["...optional balanced training list..."]
}
```

## Hyperparameters (paper Table 6)

| Hyperparameter             | Value                  |
|----------------------------|------------------------|
| Backbone                   | ResNet-18 (1-channel)  |
| Feature dim                | 512                    |
| Attention heads            | 4                      |
| Target K                   | 8                      |
| Positional encoding        | Learnable, max len 1000|
| Dropout                    | 0.1                    |
| Optimizer                  | Adam                   |
| Learning rate              | 1e-5                   |
| Batch size                 | 1 bag                  |
| Gradient accumulation      | 64 steps               |
| Gradient clip max-norm     | 1.0                    |
| Training epochs            | 50                     |
| Sparsity weights (lambda1, lambda2) | 1.0, 1.0      |
