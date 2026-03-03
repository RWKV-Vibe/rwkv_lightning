import re
import types
import torch
import os

from infer.rwkv_batch.rwkv7 import RWKV_x070
from infer.rwkv_batch.utils import TRIE_TOKENIZER


def load_model_and_tokenizer(model_path: str):
    rocm_flag = torch.version.hip is not None

    print(f"\n[INFO] Loading RWKV-7 model from {model_path}\n")

    args = types.SimpleNamespace()
    args.vocab_size = 65536
    args.head_size = 64

    args.FILE_SIZE = os.path.getsize(model_path)
    model_size = re.search(r"(?<!\w)\d+(\.\d+)?[Bb](?!\w)", model_path)
    if model_size is not None:
        args.MODEL_SIZE = float(model_size.group().replace("B", "").replace("b", ""))

    if model_path.endswith(".pth"):
        args.MODEL_NAME = re.sub(r"\.pth$", "", model_path)
    else:
        args.MODEL_NAME = model_path

    model = RWKV_x070(args)
    tokenizer = TRIE_TOKENIZER("infer/rwkv_batch/rwkv_vocab_v20230424.txt")

    print("[INFO] Model loaded successfully.\n")

    return model, tokenizer, args, rocm_flag
