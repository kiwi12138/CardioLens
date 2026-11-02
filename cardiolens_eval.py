#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import signal
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


try:
    import torch
except Exception:  # pragma: no cover
    torch = None


try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None


try:
    from vllm import LLM, SamplingParams
except Exception:  # pragma: no cover
    LLM = None
    SamplingParams = None


DEFAULT_LIMIT_MM_IMAGES = 4


@dataclass
class ModelProfile:
    name: str
    log_name: str
    default_max_tokens: int = 256
    default_max_model_len: Optional[int] = None
    default_gpu_memory_utilization: float = 0.90
    default_limit_mm_images: int = DEFAULT_LIMIT_MM_IMAGES
    default_dtype: Optional[str] = "bfloat16"
    default_model_impl: str = "auto"
    use_vllm: bool = True


PROFILE_CONFIG: Dict[str, ModelProfile] = {
    "deepseek_vl2": ModelProfile("deepseek_vl2", "deepseek_vl2"),
    "deepseek_vl2_small": ModelProfile("deepseek_vl2_small", "deepseek_vl2_small"),
    "gemma_3_27b": ModelProfile("gemma_3_27b", "gemma_3_27b", default_max_tokens=128, default_max_model_len=4096),
    "gemma_4_4b": ModelProfile("gemma_4_4b", "gemma_4_4b", default_max_tokens=128, default_max_model_len=40000),
    "gemma_4_31b": ModelProfile("gemma_4_31b", "gemma_4_31b", default_max_tokens=128, default_max_model_len=40000),
    "internvl": ModelProfile("internvl", "internvl", default_max_tokens=512),
    "llama_11b": ModelProfile("llama_11b", "llama_11b", default_max_tokens=128, default_max_model_len=4096),
    "mistral_small": ModelProfile("mistral_small", "mistral_small", default_max_tokens=64, default_limit_mm_images=10),
    "phi_4": ModelProfile("phi_4", "phi_4", default_max_tokens=128, default_max_model_len=4096),
    "fleming_38b": ModelProfile("fleming_38b", "fleming_38b", default_max_tokens=256, default_max_model_len=4096),
    "hulu_med7b": ModelProfile("hulu_med7b", "hulu_med7b", default_max_tokens=512, default_max_model_len=4096),
    "hulu_30b": ModelProfile("hulu_30b", "hulu_30b", default_max_tokens=512, default_max_model_len=4096),
    "lingshu_7b": ModelProfile("lingshu_7b", "lingshu_7b", default_max_tokens=512, default_max_model_len=4096),
    "qoqmed_7b": ModelProfile("qoqmed_7b", "qoqmed_7b", default_max_tokens=256, default_max_model_len=4096),
    "qwen_8b": ModelProfile("qwen_8b", "qwen_8b", default_max_tokens=512, default_max_model_len=4096),
    "qwen_32b_reasoning": ModelProfile("qwen_32b_reasoning", "qwen_32b_reasoning", default_max_tokens=2048, default_max_model_len=8192),
    "medgemma_4b": ModelProfile("medgemma_4b", "medgemma_4b", default_max_tokens=512, default_max_model_len=4096),
    "qwen32_medgemma27": ModelProfile("qwen32_medgemma27", "qwen32_medgemma27", default_max_tokens=512, default_max_model_len=8192),
    "lingshu_hulu": ModelProfile("lingshu_hulu", "lingshu_hulu", default_max_tokens=512, default_max_model_len=4096),
}


def setup_logger(log_file: str, logger_name: str) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass

    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    formatter = logging.Formatter("%(asctime)s [%(processName)s] %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def collect_all_json_files(root: str) -> List[str]:
    json_files: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.lower().endswith(".json"):
                json_files.append(os.path.join(dirpath, filename))
    json_files.sort()
    return json_files


def make_output_csv_path(json_path: str, json_root: str, output_root: str) -> str:
    rel_path = os.path.relpath(json_path, json_root)
    rel_csv = os.path.splitext(rel_path)[0] + ".csv"
    out_path = os.path.join(output_root, rel_csv)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    return out_path


def output_path_for_mode(path: str, mode: str) -> str:
    if mode == "path":
        return path.replace(".csv", "_path.csv")
    if mode == "pil":
        return path.replace(".csv", "_pil.csv")
    return path


def should_skip_output(output_csv_path: str, overwrite: bool) -> bool:
    return (not overwrite) and os.path.exists(output_csv_path)


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalize_image_list(image_field: Any) -> List[str]:
    if image_field is None:
        return []
    if isinstance(image_field, str):
        return [image_field]
    if isinstance(image_field, list):
        return [str(item) for item in image_field]
    return [str(image_field)]


def _resolve_image_path(image_path: str, image_root: Optional[str]) -> str:
    image_path = str(image_path).strip()
    if not image_path:
        return ""
    if os.path.isabs(image_path):
        return image_path
    if image_root:
        return os.path.join(image_root, image_path)
    return image_path


def parse_question_item(item: Dict[str, Any], image_root: Optional[str] = None) -> Tuple[str, List[str], Dict[str, Any]]:
    question_type = item.get("field", item.get("question_type", "N/A"))
    conversations = item.get("conversations", [])

    user_text = ""
    assistant_text = ""
    for message in conversations:
        if message.get("from") == "user":
            user_text = message.get("value", "")
        elif message.get("from") == "assistant":
            assistant_text = message.get("value", "")

    if not user_text:
        user_text = item.get("question", "") or item.get("query", "") or item.get("text", "")
    if not assistant_text:
        assistant_text = item.get("answer", "") or item.get("gt_answer", "") or item.get("ground_truth", "")

    images: List[str] = []
    for raw_path in _normalize_image_list(item.get("image", item.get("images", []))):
        resolved = _resolve_image_path(raw_path, image_root)
        if resolved:
            images.append(resolved)

    qid = item.get("qid", item.get("id", item.get("question_id", "")))
    meta = {
        "qid": qid,
        "field": question_type,
        "question_type": item.get("question_type", "Single Choice"),
        "is_multiple_choice": bool(item.get("is_multiple_choice", False)),
        "patient_id": item.get("patient_id", ""),
        "sequence_view": item.get("sequence_view", ""),
        "original_nii": item.get("original_nii", {}),
        "original_question": item.get("original_question", user_text),
        "task": item.get("task", ""),
        "answer": assistant_text,
        "question": user_text,
        "options": item.get("options", []),
    }
    return user_text, images, meta


def _load_image_as_rgb(path: str):
    if Image is None:
        raise RuntimeError("PIL is required to load images")
    return Image.open(path).convert("RGB")


def _load_npy_as_images(path: str, num_slices: int = 8):
    if Image is None:
        raise RuntimeError("PIL is required to load numpy images")
    import numpy as np

    data = np.load(path)
    if data.ndim == 4 and data.shape[0] == 1:
        data = data[0]
    if data.ndim != 3:
        raise ValueError(f"Unsupported npy shape: {data.shape}")
    depth = data.shape[0]
    indices = np.linspace(0, depth - 1, num_slices, dtype=int)
    images = []
    for index in indices:
        slice_img = data[index]
        slice_min = float(slice_img.min())
        slice_max = float(slice_img.max())
        if slice_max - slice_min <= 0:
            norm = slice_img * 0
        else:
            norm = (slice_img - slice_min) / (slice_max - slice_min) * 255.0
        rgb = np.stack([norm.astype("uint8")] * 3, axis=-1)
        images.append(Image.fromarray(rgb).convert("RGB"))
    return images


def load_images_for_item(image_paths: Sequence[str]) -> List[Any]:
    images: List[Any] = []
    for image_path in image_paths:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        if image_path.lower().endswith(".npy"):
            images.extend(_load_npy_as_images(image_path))
        else:
            images.append(_load_image_as_rgb(image_path))
    return images


def build_mm_content(question: str, image_inputs: Sequence[Any]) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = []
    for image_input in image_inputs:
        if isinstance(image_input, str):
            content.append({"type": "image", "image": image_input})
        else:
            content.append({"type": "image", "image": image_input})
    content.append({"type": "text", "text": question})
    return content


def _build_llm(model_path: str, gpus: List[int], args: argparse.Namespace, limit_mm_images: int):
    if LLM is None or SamplingParams is None:
        raise RuntimeError("vllm is required for CardioLens evaluation")

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in gpus)
    kwargs: Dict[str, Any] = {
        "model": model_path,
        "trust_remote_code": True,
        "tensor_parallel_size": len(gpus),
        "disable_log_stats": True,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "limit_mm_per_prompt": {"image": limit_mm_images},
    }
    if getattr(args, "dtype", None):
        kwargs["dtype"] = args.dtype
    if getattr(args, "max_model_len", None):
        kwargs["max_model_len"] = args.max_model_len
    if getattr(args, "model_impl", None):
        kwargs["model_impl"] = args.model_impl
    return LLM(**kwargs)


def _chat(llm, messages: List[Dict[str, Any]], sampling_params):
    try:
        return llm.chat(messages=messages, sampling_params=sampling_params, use_tqdm=False)
    except TypeError:
        return llm.chat(messages, sampling_params=sampling_params)


def infer_single_json_to_csv(
    *,
    model_path: str,
    json_path: str,
    output_csv_path: str,
    image_root: Optional[str],
    gpus: List[int],
    batch_size: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    limit_mm_images: int,
    max_model_len: Optional[int],
    gpu_memory_utilization: float,
    dtype: Optional[str],
    model_impl: str,
    overwrite: bool,
    logger: logging.Logger,
    start_idx: int = 0,
    end_idx: int = -1,
    skip_no_image: bool = False,
    skip_missing_images: bool = False,
    image_input_mode: str = "pil",
) -> Dict[str, Any]:
    del image_input_mode
    if should_skip_output(output_csv_path, overwrite):
        logger.info(f"[SKIP] existing output: {output_csv_path}")
        return {"json_path": json_path, "output_csv": output_csv_path, "skipped": True}

    with open(json_path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)

    if not isinstance(raw, list):
        raise ValueError(f"JSON root must be a list of samples: {json_path}")

    data = raw[start_idx:] if end_idx < 0 else raw[start_idx:end_idx]

    llm = _build_llm(model_path, gpus, argparse.Namespace(
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=dtype,
        max_model_len=max_model_len,
        model_impl=model_impl,
    ), limit_mm_images)

    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=temperature, top_p=top_p)

    written = 0
    with open(output_csv_path, "w", encoding="utf-8", newline="") as csv_file:
        fieldnames = ["qid", "field", "question_type", "is_multiple_choice", "patient_id", "sequence_view", "original_nii", "original_question", "task", "answer", "question", "options", "prediction"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for start in range(0, len(data), max(1, batch_size)):
            batch = data[start:start + max(1, batch_size)]
            messages_batch: List[List[Dict[str, Any]]] = []
            metas: List[Dict[str, Any]] = []

            for item in batch:
                question, image_paths, meta = parse_question_item(item, image_root=image_root)
                metas.append(meta)
                if not image_paths:
                    if skip_no_image:
                        writer.writerow({**meta, "prediction": ""})
                        written += 1
                        continue
                    raise ValueError(f"No images for item: {meta.get('qid', '')}")

                missing = [path for path in image_paths if not os.path.exists(path)]
                if missing:
                    if skip_missing_images:
                        writer.writerow({**meta, "prediction": f"ERROR: missing_images: {missing[:3]}"})
                        written += 1
                        continue
                    raise FileNotFoundError(f"Missing images: {missing[:3]}")

                if len(image_paths) > limit_mm_images > 0:
                    image_paths = image_paths[:limit_mm_images]

                messages_batch.append([
                    {"role": "user", "content": build_mm_content(question, image_paths)}
                ])

            if not messages_batch:
                continue

            try:
                outputs = _chat(llm, messages_batch, sampling_params)
            except Exception as exc:
                logger.error(f"vLLM chat failed for {json_path}: {exc}")
                logger.error(traceback.format_exc())
                outputs = None

            if outputs is None:
                for meta in metas:
                    writer.writerow({**meta, "prediction": "ERROR: vllm_chat_failed"})
                    written += 1
                continue

            for meta, output in zip(metas, outputs):
                prediction = ""
                try:
                    prediction = output.outputs[0].text if getattr(output, "outputs", None) else ""
                except Exception:
                    prediction = str(output)
                writer.writerow({**meta, "prediction": prediction})
                written += 1

    logger.info(f"CSV written: {output_csv_path} | rows={written}")
    return {"json_path": json_path, "output_csv": output_csv_path, "rows": written, "skipped": False}


def run_single_profile(args: argparse.Namespace, profile: ModelProfile) -> None:
    if not os.path.isdir(args.json_path) and not args.output_csv_path and not args.output_root:
        raise ValueError("Need output_csv_path or output_root")

    if os.path.isdir(args.json_path):
        json_root = os.path.abspath(args.json_path)
        json_files = collect_all_json_files(json_root)
        if not json_files:
            raise FileNotFoundError(f"No .json files found under: {args.json_path}")
        if not args.output_root:
            raise ValueError("When json_path is a directory, output_root is required")
        if args.output_csv_path:
            raise ValueError("When json_path is a directory, output_csv_path is not allowed")
        if args.json_root is None:
            args.json_root = json_root
    else:
        json_files = [args.json_path]
        if not args.output_csv_path:
            if not args.output_root or not args.json_root:
                raise ValueError("Single file mode requires output_csv_path or both json_root and output_root")
            args.output_csv_path = make_output_csv_path(args.json_path, args.json_root, args.output_root)

    modes = ["pil", "path"] if args.image_input_mode == "both" else [args.image_input_mode]
    for mode in modes:
        for json_file in json_files:
            if len(json_files) == 1 and args.output_csv_path:
                out_csv = output_path_for_mode(args.output_csv_path, mode) if len(modes) > 1 else args.output_csv_path
            else:
                out_csv = output_path_for_mode(make_output_csv_path(json_file, args.json_root, args.output_root), mode) if len(modes) > 1 else make_output_csv_path(json_file, args.json_root, args.output_root)
            infer_single_json_to_csv(
                model_path=args.model_path,
                json_path=json_file,
                output_csv_path=out_csv,
                image_root=args.image_root,
                gpus=args.gpus,
                batch_size=args.batch_size,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                limit_mm_images=args.limit_mm_images,
                max_model_len=args.max_model_len,
                gpu_memory_utilization=args.gpu_memory_utilization,
                dtype=args.dtype,
                model_impl=args.model_impl,
                overwrite=args.overwrite,
                logger=args.logger,
                start_idx=args.start_idx,
                end_idx=args.end_idx,
                skip_no_image=args.skip_no_image,
                skip_missing_images=args.skip_missing_images,
                image_input_mode=mode,
            )


def build_arg_parser(description: str = "CardioLens evaluation") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--json_root", type=str, default=None)
    parser.add_argument("--image_root", type=str, default=None)
    parser.add_argument("--output_csv_path", type=str, default=None)
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--log_file", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--model_impl", type=str, default="auto")
    parser.add_argument("--limit_mm_images", type=int, default=DEFAULT_LIMIT_MM_IMAGES)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--skip_no_image", action="store_true")
    parser.add_argument("--skip_missing_images", action="store_true")
    parser.add_argument("--image_input_mode", type=str, default="pil", choices=["pil", "path", "both"])
    parser.add_argument("--overwrite", action="store_true")
    return parser


def resolve_log_file(args: argparse.Namespace, profile: ModelProfile) -> str:
    if args.log_file:
        return args.log_file
    log_dir = args.log_dir or os.path.join(os.getcwd(), "logs")
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, f"{profile.log_name}.log")


def run_profile(profile_name: str, args: argparse.Namespace) -> None:
    profile = PROFILE_CONFIG[profile_name]
    args.log_file = resolve_log_file(args, profile)
    args.logger = setup_logger(args.log_file, f"cardiolens_{profile_name}")

    signal.signal(signal.SIGINT, lambda signum, frame: (_ for _ in ()).throw(SystemExit(128 + signum)))
    signal.signal(signal.SIGTERM, lambda signum, frame: (_ for _ in ()).throw(SystemExit(128 + signum)))

    args.logger.info(f"profile={profile_name}")
    args.logger.info(f"model_path={args.model_path}")
    args.logger.info(f"json_path={args.json_path}")
    args.logger.info(f"json_root={args.json_root}")
    args.logger.info(f"output_root={args.output_root}")
    args.logger.info(f"output_csv_path={args.output_csv_path}")
    args.logger.info(f"gpus={args.gpus}")

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model path not found: {args.model_path}")
    if not os.path.exists(args.json_path):
        raise FileNotFoundError(f"JSON path not found: {args.json_path}")

    run_single_profile(args, profile)
