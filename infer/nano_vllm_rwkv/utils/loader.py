import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    # Check for RWKV pth file first
    if os.path.isfile(path) and path.endswith(".pth"):
        pth_files = [path]
    else:
        pth_files = glob(os.path.join(path, "*.pth"))
    if pth_files and hasattr(model, "load_pth"):
        # Use model-specific pth loading (e.g., for RWKV)
        model.load_pth(pth_files[0])
        return

    if os.path.isfile(path):
        raise ValueError(f"Unsupported model file path: {path}")

    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
