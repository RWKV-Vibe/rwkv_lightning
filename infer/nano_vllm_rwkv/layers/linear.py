import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import sys
import os
import ctypes
import sysconfig
from pathlib import Path
from functools import lru_cache

def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


@lru_cache(maxsize=1)
def _preload_env_cuda_libs() -> list[str]:
    purelib = Path(sysconfig.get_paths().get("purelib", ""))
    nvidia_root = purelib / "nvidia"
    if not nvidia_root.exists():
        return []

    loaded: list[str] = []
    lib_roots = [
        nvidia_root / "cuda_runtime" / "lib",
        nvidia_root / "cuda_nvrtc" / "lib",
        nvidia_root / "cu13" / "lib",
        nvidia_root / "cublas" / "lib",
        nvidia_root / "cusparselt" / "lib",
        nvidia_root / "cusparse" / "lib",
        nvidia_root / "cusolver" / "lib",
        nvidia_root / "nccl" / "lib",
    ]
    candidates = [
        nvidia_root / "cuda_runtime" / "lib" / "libcudart.so.12",
        nvidia_root / "cuda_nvrtc" / "lib" / "libnvrtc.so.12",
        nvidia_root / "cuda_nvrtc" / "lib" / "libnvrtc-builtins.so.12.8",
        nvidia_root / "cu13" / "lib" / "libnvrtc.so.13",
        nvidia_root / "cu13" / "lib" / "libnvrtc-builtins.so.13.0",
        nvidia_root / "cublas" / "lib" / "libcublasLt.so.12",
        nvidia_root / "cublas" / "lib" / "libcublas.so.12",
        nvidia_root / "cublas" / "lib" / "libcublasLt.so.13",
        nvidia_root / "cublas" / "lib" / "libcublas.so.13",
        nvidia_root / "cusparselt" / "lib" / "libcusparseLt.so.0",
        nvidia_root / "cusparse" / "lib" / "libcusparse.so.12",
        nvidia_root / "cusolver" / "lib" / "libcusolver.so.11",
        nvidia_root / "nccl" / "lib" / "libnccl.so.2",
    ]
    for lib in candidates:
        if not lib.exists():
            continue
        ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
        loaded.append(str(lib))
    return loaded


@lru_cache(maxsize=1)
def _get_marlin_impl():
    try:
        _preload_env_cuda_libs()
        from vllm.model_executor.layers.quantization.rtn import rtn_quantize, repack_weights
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            apply_rtn_marlin_linear,
            marlin_make_workspace_new,
        )
        from vllm.scalar_type import scalar_types
        return {
            "rtn_quantize": rtn_quantize,
            "repack_weights": repack_weights,
            "apply_rtn_marlin_linear": apply_rtn_marlin_linear,
            "marlin_make_workspace_new": marlin_make_workspace_new,
            "scalar_types": scalar_types,
        }
    except Exception:
        return None


def get_marlin_impl_or_raise():
    marlin = _get_marlin_impl()
    if marlin is not None:
        return marlin
    current_python = Path(sys.executable).resolve()
    current_prefix = current_python.parent.parent
    site_packages = next((p for p in sys.path if "site-packages" in p), "")
    cuda_lib_root = Path(site_packages) / "nvidia" if site_packages else None
    cuda_libs = []
    if cuda_lib_root is not None and cuda_lib_root.exists():
        for rel in (
            "cuda_runtime/lib",
            "cu13/lib",
            "cuda_nvrtc/lib",
            "cublas/lib",
            "cusparse/lib",
            "cusparselt/lib",
            "cusolver/lib",
            "nccl/lib",
        ):
            lib_path = cuda_lib_root / rel
            if lib_path.exists():
                cuda_libs.append(str(lib_path))
    ld_hint = ":".join(cuda_libs) if cuda_libs else "<python-env>/lib/pythonX.Y/site-packages/nvidia/.../lib"
    raise RuntimeError(
        "Marlin runtime is unavailable. rwkv_quant_int8 now requires Marlin and no longer falls back. "
        f"Current python: {current_python}. Current prefix: {current_prefix}. "
        "Use the Python interpreter from the environment where vLLM/Marlin is installed. "
        "The loader already tries to preload CUDA libs from the active environment; if that still fails, set "
        f"LD_LIBRARY_PATH to include that environment's NVIDIA CUDA runtime libraries, e.g. {ld_hint}. "
        f"Current LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH', '')!r}"
    )


class LinearBase(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
    ):
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReplicatedLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class MatmulLinear(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        weight_layout: str = "in_out",
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        assert weight_layout in ("in_out", "out_in")
        self.weight_layout = weight_layout
        if weight_layout == "in_out":
            self.weight = nn.Parameter(torch.empty(input_size, output_size))
        else:
            self.weight = nn.Parameter(torch.empty(output_size, input_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight_layout == "in_out":
            return F.linear(x, self.weight.t(), self.bias)
        return F.linear(x, self.weight, self.bias)


def _int8_cublas_dequant_eager(
    y_int32: torch.Tensor,
    x_scale: torch.Tensor,
    scales_fp16: torch.Tensor,
) -> torch.Tensor:
    return y_int32.to(torch.float16) * x_scale.to(torch.float16) * scales_fp16


def _int8_cublas_quant_eager(
    x: torch.Tensor,
    act_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_absmax = x.abs().amax(dim=1, keepdim=True).clamp(min=1e-5)
    x_scale = x_absmax / act_scale
    x_int8 = torch.clamp(torch.round(x / x_scale), -128, 127).to(torch.int8)
    return x_int8, x_scale


try:
    _int8_cublas_dequant = torch.compile(_int8_cublas_dequant_eager, dynamic=True)
except Exception:
    _int8_cublas_dequant = _int8_cublas_dequant_eager

def _int8_per_channel_cublas_impl(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    scales_fp16: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    act_scale: float = 127.0,
) -> torch.Tensor:
    orig_shape = x.shape[:-1]
    k = x.shape[-1]
    x2 = x.reshape(-1, k)
    assert qweight.dim() == 2
    n, k2 = qweight.shape
    assert k == k2
    if scales_fp16 is None:
        scales_fp16 = scales.to(torch.float16)
    if x2.shape[0] <= 16:
        # torch._int_mm performs well for lm_head, but very small M can require
        # padding. Avoid materializing a full fp16 dequantized weight matrix
        # (~512 MiB for 7B vocab heads), which breaks bs=1 ping-pong graph use.
        min_intmm_rows = 17
        x_pad = torch.zeros((min_intmm_rows, k), device=x2.device, dtype=x2.dtype)
        x_pad[:x2.shape[0]].copy_(x2)
        x_int8, x_scale = _int8_cublas_quant(x_pad, act_scale)
        y_int32 = torch._int_mm(x_int8, qweight.t())
        y = (
            y_int32.to(torch.float32)
            * x_scale.to(torch.float32)
            * scales_fp16.to(torch.float32)
        ).to(torch.float16)[:x2.shape[0]]
    else:
        x_int8, x_scale = _int8_cublas_quant(x2, act_scale)
        y_int32 = torch._int_mm(x_int8, qweight.t())
        y = _int8_cublas_dequant(y_int32, x_scale, scales_fp16)
    if bias is not None:
        y = y + bias.to(torch.float16)
    y = y.to(x.dtype)
    return y.reshape(*orig_shape, n)


def _int8_per_channel_cublas(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    scales_fp16: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    assert hasattr(torch, "_int_mm")
    assert x.is_cuda and qweight.is_cuda and scales.is_cuda
    assert x.dtype in (torch.float16, torch.bfloat16)
    return _int8_per_channel_cublas_impl(x, qweight, scales, scales_fp16, bias)


try:
    _int8_cublas_quant = torch.compile(_int8_cublas_quant_eager, dynamic=True)
except Exception:
    _int8_cublas_quant = _int8_cublas_quant_eager


class MarlinInt8Linear(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        group_size: int = 128,
        scale_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.group_size = group_size
        self.scale_dtype = scale_dtype
        self.register_buffer("qweight", None)
        self.register_buffer("scales", None)
        self.register_buffer("workspace", None)
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
        else:
            self.register_parameter("bias", None)

    @torch.no_grad()
    def quantize_from_weight(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        *,
        weight_layout: str = "auto",
    ):
        marlin = get_marlin_impl_or_raise()
        assert self.group_size == 128, "Minimal Marlin experiment only supports group_size=128."
        assert weight_layout in ("auto", "in_out", "out_in")
        assert weight.shape in (
            (self.input_size, self.output_size),
            (self.output_size, self.input_size),
        )
        # vLLM RTN/Marlin expects row-major [out, in]. Square projections make
        # shape-based inference ambiguous, so prefer the explicit module layout
        # when available.
        if weight_layout == "in_out":
            weight_oi = weight.t().contiguous()
        elif weight_layout == "out_in":
            weight_oi = weight.contiguous()
        else:
            weight_oi = weight if weight.shape == (self.output_size, self.input_size) else weight.t().contiguous()
        q_u8, scales = marlin["rtn_quantize"](weight_oi, 8, self.group_size)
        q_packed, s_packed = marlin["repack_weights"](q_u8, scales, 8)
        self.qweight = q_packed.contiguous()
        self.scales = s_packed.to(self.scale_dtype).contiguous()
        self.workspace = marlin["marlin_make_workspace_new"](weight.device, 4)
        if self.bias is not None and bias is not None:
            self.bias.data.copy_(bias)
        elif self.bias is not None:
            self.bias.data.zero_()
        return self

    @classmethod
    @torch.no_grad()
    def from_float(cls, module: MatmulLinear):
        qmod = cls(
            module.input_size,
            module.output_size,
            bias=module.bias is not None,
            group_size=128,
            scale_dtype=module.weight.dtype if module.weight.dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.float16,
        )
        bias = None if module.bias is None else module.bias.detach()
        qmod.quantize_from_weight(
            module.weight.detach(),
            bias,
            weight_layout=getattr(module, "weight_layout", "auto"),
        )
        return qmod

    @property
    def weight(self):
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        marlin = get_marlin_impl_or_raise()
        orig_shape = x.shape[:-1]
        x2 = x.reshape(-1, x.shape[-1])
        y = marlin["apply_rtn_marlin_linear"](
            input=x2,
            weight=self.qweight,
            weight_scale=self.scales,
            workspace=self.workspace,
            quant_type=marlin["scalar_types"].uint8b128,
            output_size_per_partition=self.output_size,
            input_size_per_partition=self.input_size,
            bias=self.bias,
        )
        return y.reshape(*orig_shape, self.output_size)


class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
