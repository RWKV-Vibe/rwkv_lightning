import re
import types
import torch

from infer.rwkv_batch.utils import TRIE_TOKENIZER


def load_model_and_tokenizer(model_path: str, w8a8: bool = False):
    rocm_flag = torch.version.hip is not None

    quant_mode = "W8A8 int8" if w8a8 else "FP16"
    print(f"\n[INFO] Loading RWKV-7 {quant_mode} model from {model_path}\n")

    args = types.SimpleNamespace()
    args.vocab_size = 65536
    args.head_size = 64
    args.w8a8 = w8a8
    if model_path.endswith(".pth"):
        args.MODEL_NAME = re.sub(r"\.pth$", "", model_path)
    else:
        args.MODEL_NAME = model_path

    if w8a8:
        from infer.rwkv_batch.rwkv7.modeling_rwkv7_int8 import RWKV_x070
    else:
        from infer.rwkv_batch.rwkv7.modeling_rwkv7 import RWKV_x070

    model = RWKV_x070(args)
    tokenizer = TRIE_TOKENIZER("infer/rwkv_batch/rwkv_vocab_v20230424.txt")

    print(f"[INFO] {quant_mode} model loaded successfully.\n")

    return model, tokenizer, args, rocm_flag
