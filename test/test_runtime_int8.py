#!/usr/bin/env python3

import argparse
import os
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0;8.6;9.0;12.0")
os.environ.setdefault("RWKV_USE_COMPILE", "0")

from model_load.model_loader import load_model_and_tokenizer


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test RWKV int8 runtime")
    parser.add_argument(
        "--model-path",
        default="/models/rwkv7-g1f-7.2b-20260414-ctx8192",
        help="Model path without or with .pth suffix",
    )
    parser.add_argument("--prompt", default="Hello from int8")
    args = parser.parse_args()

    start = time.perf_counter()
    model, tokenizer, model_args, _ = load_model_and_tokenizer(args.model_path, runtime="int8")
    load_s = time.perf_counter() - start

    state = model.generate_zero_state(1)
    tokens = [tokenizer.encode(args.prompt)]

    infer_start = time.perf_counter()
    out = model.forward_batch(tokens, state)
    infer_s = time.perf_counter() - infer_start

    expected_shape = (1, model_args.vocab_size)
    actual_shape = tuple(out.shape)
    if actual_shape != expected_shape:
        raise SystemExit(
            f"Unexpected output shape for int8 runtime: {actual_shape}, expected {expected_shape}"
        )

    print(f"int8 load_s={load_s:.2f}")
    print(f"int8 infer_s={infer_s:.2f}")
    print(f"int8 output_shape={actual_shape}")
    print("int8 runtime ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
