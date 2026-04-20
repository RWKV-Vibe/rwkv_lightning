import os
import re
import types

import torch

from infer.rwkv_batch.utils import TRIE_TOKENIZER


def _resolve_runtime(runtime: str) -> str:
    normalized = (runtime or "fp16").strip().lower()
    if normalized not in {"fp16", "int8"}:
        raise ValueError(f"Unsupported runtime '{runtime}'. Expected one of: fp16, int8")
    return normalized


def _load_model_class(runtime: str):
    if runtime == "int8":
        from infer.rwkv_batch.rwkv7_int8 import RWKV_x070
    else:
        from infer.rwkv_batch.rwkv7 import RWKV_x070

    return RWKV_x070


def _build_model_metadata(model_path: str, runtime: str) -> dict:
    model_root = re.sub(r"\.pth$", "", model_path)
    model_name = os.path.basename(model_root)
    served_model_id = f"{model_name}:{runtime}"
    aliases = [served_model_id, model_name, "rwkv7"]
    return {
        "model_root": model_root,
        "model_name": model_name,
        "served_model_id": served_model_id,
        "runtime": runtime,
        "aliases": tuple(dict.fromkeys(aliases)),
    }


def load_model_and_tokenizer(model_path: str, runtime: str = "fp16"):
    rocm_flag = torch.version.hip is not None
    runtime = _resolve_runtime(runtime)
    metadata = _build_model_metadata(model_path, runtime)

    print(
        f"\n[INFO] Loading RWKV-7 model from {metadata['model_root']} "
        f"(runtime={runtime})\n"
    )

    args = types.SimpleNamespace()
    args.vocab_size = 65536
    args.head_size = 64
    args.MODEL_NAME = metadata["model_root"]
    args.runtime = runtime
    args.model_name = metadata["model_name"]
    args.served_model_id = metadata["served_model_id"]
    args.model_aliases = metadata["aliases"]

    model_class = _load_model_class(runtime)
    model = model_class(args)
    tokenizer = TRIE_TOKENIZER("infer/rwkv_batch/rwkv_vocab_v20230424.txt")

    print(
        f"[INFO] Model loaded successfully as {args.served_model_id} "
        f"(aliases: {', '.join(args.model_aliases)}).\n"
    )

    return model, tokenizer, args, rocm_flag
