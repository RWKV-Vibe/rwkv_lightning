import re
import types
import torch

from infer.rwkv_batch.utils import TRIE_TOKENIZER


def load_model_and_tokenizer(model_path: str, pp_devices=None):
    rocm_flag = torch.version.hip is not None

    print(f"\n[INFO] Loading RWKV-7 model from {model_path}\n")

    args = types.SimpleNamespace()
    args.vocab_size = 65536
    args.head_size = 64
    if model_path.endswith(".pth"):
        args.MODEL_NAME = re.sub(r"\.pth$", "", model_path)
    else:
        args.MODEL_NAME = model_path

    args.pp_devices = list(pp_devices) if pp_devices else None
    args.use_pp = bool(args.pp_devices)

    if args.use_pp:
        from infer.rwkv_batch.rwkv7_pp import RWKV_x070

        model = RWKV_x070(args, devices=args.pp_devices)
    else:
        from infer.rwkv_batch.rwkv7 import RWKV_x070

        model = RWKV_x070(args)
    tokenizer = TRIE_TOKENIZER("infer/rwkv_batch/rwkv_vocab_v20230424.txt")

    print("[INFO] Model loaded successfully.\n")

    return model, tokenizer, args, rocm_flag
