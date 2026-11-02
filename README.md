# CardioLens

CardioLens is a benchmark and evaluation toolkit for multi-sequence cardiac MRI, accompanying the paper *CardioLens: Revealing the Clinical Reality Gap of MLLMs via Multi-Sequence Cardiac MRI Evaluations*. The paper's core message is that modern medical MLLMs may look strong on public 2D benchmarks, but still show a clear clinical reality gap on realistic, patient-level CMR studies.

The repository provides:

1. **Evaluation code** (`cardiolens_eval.py`, `run_models.py`) for sending patient-level JSON samples and image inputs through vLLM and writing predictions to CSV.
2. **Task-Aware MIL code** (`mil_code/`) for data-driven slice selection via Multiple Instance Learning, used as one of the slice-selection protocols described in the paper.

The evaluation supports the three task tiers described in the paper:

1. Image understanding: sequence-level clinical finding recognition.
2. Report generation: converting structured findings into report-style text.
3. Disease diagnosis: patient-level diagnosis from multi-sequence evidence.

## Repository Layout

```
CardioLens/
├── cardiolens_eval.py              # Core inference and evaluation logic
├── run_models.py                   # Unified launcher for all model profiles
├── examples/
│   └── redacted_lge_sample.json    # Redacted example of the expected data format
├── requirement-env1.txt             # Dependencies for env1 (most models)
├── requirement-env2.txt             # Dependencies for env2 (Gemma4 only)
├── mil_code/                       # Task-Aware MIL for data-driven slice selection
│   ├── train.py                    # Train the MIL model
│   ├── evaluate.py                 # Evaluate a trained MIL checkpoint
│   ├── extract_topk.py             # Extract top-K slices for MLLM evaluation
│   ├── README.md                   # MIL-specific documentation
│   └── src/
│       ├── __init__.py
│       ├── model.py                # Encoder + per-task transformer attention + classifier
│       ├── dataset.py              # Multi-sequence cardiac MRI dataset
│       ├── losses.py               # Multi-task loss + target-K sparsity regularization
│       ├── metrics.py              # Acc / Precision / Recall / F1 / EM
│       └── trainer.py              # Training loop
└── README.md                       # This file
```

### MLLM Evaluation Scripts

- [cardiolens_eval.py](cardiolens_eval.py): core inference and evaluation logic.
- [run_models.py](run_models.py): unified launcher supporting all model profiles (e.g. `hulu_med7b`, `qwen_8b`, `lingshu_hulu`, `qwen32_medgemma27`).

### Task-Aware MIL (Data-Driven Slice Selection)

The `mil_code/` directory contains the implementation of the Task-Aware Multiple Instance Learning (MIL) framework described in the paper (Section 2.4 / Appendix C). This model learns per-slice, per-task importance weights from sequence-level supervision, and its top-K outputs serve as inputs for the data-driven MIL sampling protocol used in MLLM evaluation.

Key components:
- **`train.py`**: Train the MIL model with multi-task loss and sparsity regularization.
- **`evaluate.py`**: Evaluate a trained checkpoint, reporting per-task Accuracy, Precision, Recall, F1, and Exact-match.
- **`extract_topk.py`**: Extract top-K slice indices (both per-task and task-averaged) for downstream MLLM evaluation.

See [`mil_code/README.md`](mil_code/README.md) for detailed usage instructions and alignment with the paper.

### Other Files

- [examples/redacted_lge_sample.json](examples/redacted_lge_sample.json): redacted example of the expected data format.
- [requirement-env1.txt](requirement-env1.txt): dependencies for env1 (most models).
- [requirement-env2.txt](requirement-env2.txt): dependencies for env2 (Gemma4 only).

## Environment Setup

During evaluation, we found that different models depend on different versions of the `transformers` library. We use the most compatible versions possible, but Gemma4 — a newer model — requires a higher version of `transformers` that cannot coexist with the other models in the same environment. Therefore, two separate Conda environments are needed:

- **`env1`** (`vlmqwen`): supports all models **except** Gemma4. Uses `requirement-env1.txt`.
- **`env2`** (`vlmgemma4`): supports Gemma4-based models only. Uses `requirement-env2.txt`.

Install a PyTorch wheel that matches your CUDA version first, then install the requirements.

### env1 — Most models

```bash
cd CardioLens
conda create -n env1 python=3.11 -y
conda activate env1
pip install --index-url https://download.pytorch.org/whl/cu118 \
  torch==2.2.0+cu118 torchvision==0.16.0+cu118 --extra-index-url https://pypi.org/simple
pip install -r requirement-env1.txt
```

### env2 — Gemma4 only

```bash
conda create -n env2 python=3.11 -y
conda activate env2
pip install --index-url https://download.pytorch.org/whl/cu118 \
  torch==2.2.0+cu118 torchvision==0.16.0+cu118 --extra-index-url https://pypi.org/simple
pip install -r requirement-env2.txt
```

### MIL training environment

```bash
conda create -n mil python=3.11 -y
conda activate mil
pip install torch torchvision tqdm numpy
```

Note: evaluation depends on vLLM and GPU inference. CPU mode may work for limited testing, but it is far too slow for practical evaluation.

## Data Format

The evaluation scripts read a JSON array, where each element is one sample. The code prefers `conversations` for the user prompt and reference answer, but can fall back to `question`, `query`, `text`, and `answer` when needed. `image` may be a single path or a list of paths; if the paths are relative, the script resolves them using `--image_root`.

Example:

```json
{
  "qid": "patient0067::enhancement_status::mcq_1",
  "field": "Enhancement Status",
  "image": [
    "Images/patient0067_LGE_sax_slice_00.png",
    "Images/patient0067_LGE_sax_slice_01.png",
    "Images/patient0067_LGE_sax_slice_02.png",
    "Images/patient0067_LGE_sax_slice_03.png",
    "Images/patient0067_LGE_sax_slice_04.png",
    "Images/patient0067_LGE_sax_slice_05.png",
    "Images/patient0067_LGE_sax_slice_06.png",
    "Images/patient0067_LGE_sax_slice_07.png"
  ],
  "patient_id": "patient0067",
  "question_type": "Single Choice",
  "is_multiple_choice": false,
  "conversations": [
    {
      "from": "user",
      "value": "You are a Vision-Language Model (VLM) for cardiac LGE MRI.\nTask: Using the provided image frames, decide whether abnormal delayed enhancement is present (single-choice).\n\nStrict rules:\n- Choose exactly ONE option: A or B.\n- No explanations, reasoning, descriptions, confidence, or extra characters.\n- Even if uncertain, choose the most likely answer.\n\nOptions:\nA. No Abnormal Delayed Enhancement\nB. Abnormal Delayed Enhancement"
    },
    {
      "from": "assistant",
      "value": "B. Abnormal Delayed Enhancement"
    }
  ]
}
```

Each JSON file is a sample list: each item binds a set of slices, a question, a reference answer, and metadata. The evaluation writes one CSV row per sample.

## Running the Code

### MLLM Evaluation

`run_models.py` is the unified entrypoint for all model profiles. It uses the selected `--profile` to fill in default values such as `max_tokens`, `max_model_len`, `dtype`, and `limit_mm_images`.

```bash
# Single-model evaluation (env1)
conda activate env1
python run_models.py \
  --profile hulu_med7b \
  --json_path ./examples/redacted_lge_sample.json \
  --json_root ./examples \
  --output_root ./outputs \
  --model_path /path/to/local_model \
  --image_root ./examples/images \
  --log_dir ./logs \
  --batch_size 1

# Paired-model evaluation (env1)
python run_models.py \
  --profile lingshu_hulu \
  --json_path ./examples/redacted_lge_sample.json \
  --json_root ./examples \
  --output_root ./outputs \
  --model_path /path/to/local_model \
  --image_root ./examples/images \
  --log_dir ./logs

# Gemma4 evaluation (env2)
conda activate env2
python run_models.py \
  --profile gemma_4_31b \
  --json_path ./examples/redacted_lge_sample.json \
  --json_root ./examples \
  --output_root ./outputs \
  --model_path /path/to/local_model \
  --image_root ./examples/images \
  --log_dir ./logs
```

You can also pass `--output_csv_path` directly if you already know the exact output file. Keep these rules in mind:

- If `--json_path` points to a single file, use either `--output_csv_path` or `--json_root` + `--output_root`.
- If `--json_path` points to a directory, you must use `--output_root`, and `--output_csv_path` is not allowed.

### Task-Aware MIL Training & Slice Extraction

The MIL model is used for the data-driven slice selection protocol described in the paper. It learns per-slice, per-task importance weights from sequence-level image-understanding labels.

```bash
# 1. Train the MIL model
python mil_code/train.py \
  --cache-dir /path/to/preprocessed_cache \
  --label-json /path/to/labels.json \
  --checkpoint-dir ./checkpoints

# 2. Evaluate the trained checkpoint
python mil_code/evaluate.py \
  --checkpoint ./checkpoints/best_model_acc.pth \
  --cache-dir /path/to/preprocessed_cache \
  --label-json /path/to/labels.json \
  --output ./val_metrics.json

# 3. Extract top-K slices for downstream MLLM evaluation
python mil_code/extract_topk.py \
  --checkpoint ./checkpoints/best_model_acc.pth \
  --cache-dir /path/to/preprocessed_cache \
  --label-json /path/to/labels.json \
  --output ./topk_indices.json \
  --target-k 8
```

The `extract_topk.py` output provides both per-task selections (for image understanding evaluation) and task-averaged selections (for report generation and disease diagnosis). See [`mil_code/README.md`](mil_code/README.md) for full details.

## Common Arguments

- `--profile`: model profile, taken from `PROFILE_CONFIG` in [cardiolens_eval.py](cardiolens_eval.py).
- `--json_path`: a single JSON file or a JSON directory.
- `--json_root`: root directory used to build output paths.
- `--output_csv_path`: direct output path for single-file mode.
- `--output_root`: required for directory mode; output mirrors the JSON directory structure.
- `--image_root`: root directory used to resolve relative image paths.
- `--gpus`: for example, `--gpus 0 1`.
- `--batch_size`: inference batch size.
- `--limit_mm_images`: maximum number of images per sample; extra images are truncated.
- `--image_input_mode`: `pil`, `path`, or `both`.
- `--skip_no_image` / `--skip_missing_images`: skip problematic samples instead of failing.
- `--overwrite`: allow overwriting existing outputs.

If `--image_input_mode both` is used, the script runs both input modes and writes `_pil.csv` and `_path.csv` outputs.

## Available Profiles

`run_models.py` supports **all** profiles defined in `PROFILE_CONFIG`:

**env1 models** (use `requirement-env1.txt`):
`deepseek_vl2`, `deepseek_vl2_small`, `gemma_3_27b`, `internvl`, `llama_11b`, `mistral_small`, `phi_4`, `fleming_38b`, `hulu_med7b`, `hulu_30b`, `lingshu_7b`, `qoqmed_7b`, `qwen_8b`, `qwen_32b_reasoning`, `medgemma_4b`, `lingshu_hulu`, `qwen32_medgemma27`

**env2 models** (use `requirement-env2.txt`):
`gemma_4_4b`, `gemma_4_31b`

## Minimal Sanity Check

To verify the pipeline end to end, run the example JSON through a small batch and confirm that the CSV and logs are generated.

```bash
conda activate env1
python run_models.py \
  --profile hulu_med7b \
  --json_path ./examples/redacted_lge_sample.json \
  --json_root ./examples \
  --output_csv_path ./outputs/sample.csv \
  --model_path /path/to/local_model \
  --image_root ./examples/images \
  --log_dir ./logs \
  --batch_size 1 \
  --overwrite
```

If you have more JSON files, point `--json_path` at the directory. The script will recursively process all `.json` files and preserve the same relative structure under `--output_root`.

