#!/usr/bin/env python3
"""
Unified CardioLens evaluation launcher.

Supports ALL model profiles defined in cardiolens_eval.PROFILE_CONFIG,
including both single-model profiles (e.g. hulu_med7b, qwen_8b) and
paired-model profiles (e.g. lingshu_hulu, qwen32_medgemma27).

Usage examples:

  # Single model
  python run_models.py --profile hulu_med7b \
      --json_path ./examples/redacted_lge_sample.json \
      --model_path /path/to/model --image_root ./examples/images

  # Paired model (e.g. lingshu + hulu ensemble)
  python run_models.py --profile lingshu_hulu \
      --json_path ./examples/redacted_lge_sample.json \
      --model_path /path/to/model --image_root ./examples/images

  # Any other profile
  python run_models.py --profile qwen32_medgemma27 \
      --json_path ./examples/redacted_lge_sample.json \
      --model_path /path/to/model --image_root ./examples/images
"""
from __future__ import annotations

import argparse
from cardiolens_eval import PROFILE_CONFIG, build_arg_parser, run_profile


def main() -> None:
    parser = build_arg_parser("CardioLens evaluation launcher")
    parser.add_argument(
        "--profile",
        type=str,
        required=True,
        choices=sorted(PROFILE_CONFIG.keys()),
        help="Model profile name, e.g. hulu_med7b, qwen_8b, lingshu_hulu, qwen32_medgemma27.",
    )
    args = parser.parse_args()

    profile = PROFILE_CONFIG[args.profile]
    if args.max_tokens is None:
        args.max_tokens = profile.default_max_tokens
    if args.max_model_len is None:
        args.max_model_len = profile.default_max_model_len
    if args.dtype is None:
        args.dtype = profile.default_dtype
    if args.limit_mm_images is None:
        args.limit_mm_images = profile.default_limit_mm_images

    run_profile(args.profile, args)


if __name__ == "__main__":
    main()