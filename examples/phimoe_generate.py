from __future__ import annotations

import argparse

import torch
from transformers import AutoTokenizer

import sys
import os
sys.path.append(os.path.abspath("."))
from moe_runtime import OffloadConfig, load_offloaded_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phi-tiny-MoE with expert offload caches.")
    parser.add_argument("--model-path", default="models/Phi-tiny-MoE-instruct")
    parser.add_argument("--prompt", default="Write a short note about MoE expert offloading.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--cpu-cache", default="4GiB")
    parser.add_argument("--gpu-cache", default="1GiB")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--attn-implementation", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    parser.add_argument("--no-prefetch", action="store_true")
    args = parser.parse_args()

    config = OffloadConfig(
        model_path=args.model_path,
        device=args.device,
        dtype=args.dtype,
        cpu_cache_bytes=args.cpu_cache,
        gpu_cache_bytes=args.gpu_cache,
        prefetch=not args.no_prefetch,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = load_offloaded_model(args.model_path, config, attn_implementation=args.attn_implementation)

    inputs = tokenizer(args.prompt, return_tensors="pt").to(config.device)
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)

    print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
    print("\n[moe-offload stats]")
    for key, value in model._moe_offload_manager.stats().items():
        print(f"{key}: {value}")
    if config.device.type == "cuda":
        print(f"cuda_peak_allocated_bytes: {torch.cuda.max_memory_allocated(config.device)}")


if __name__ == "__main__":
    main()
