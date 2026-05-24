import re
import types
import torch

from infer.rwkv_batch.rwkv7_v3a import RWKV_x070
from infer.rwkv_batch.utils import TRIE_TOKENIZER


def load_model_and_tokenizer(
    model_path: str,
    wkv_mode: str = "fp16",
    emb_device: str = "cpu",
    pp_devices=None,
    use_cuda_graph: bool = False,
):
    rocm_flag = torch.version.hip is not None

    print(f"\n[INFO] Loading RWKV-7 model from {model_path}\n")

    args = types.SimpleNamespace()
    args.vocab_size = 65536
    args.head_size = 64
    args.WKV_MODE = wkv_mode
    args.EMB_DEVICE = emb_device
    args.PP_DEVICES = pp_devices
    args.USE_CUDA_GRAPH = use_cuda_graph
    if model_path.endswith(".pth"):
        args.MODEL_NAME = re.sub(r"\.pth$", "", model_path)
    else:
        args.MODEL_NAME = model_path

    model = RWKV_x070(args)
    tokenizer = TRIE_TOKENIZER("infer/rwkv_batch/rwkv_vocab_v20230424.txt")

    print("[INFO] Model loaded successfully.\n")

    return model, tokenizer, args, rocm_flag
