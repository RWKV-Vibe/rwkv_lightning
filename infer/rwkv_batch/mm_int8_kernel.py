import torch
import os
from torch.nn import functional as F
from torch.utils.cpp_extension import load
current_path = os.path.dirname(os.path.abspath(__file__))

DTYPE = torch.half
ROCm_flag = torch.version.hip is not None
MyStatic = torch.jit.script 


if ROCm_flag == True:
    load(name="rwkv_mm8", sources=[f"{current_path}/hip/wrapper.cpp", 
                                   f"{current_path}/hip/operators.hip", 
                                   f"{current_path}/hip/gemm_fp16_cublas.cpp"], is_python_module=False,
                    verbose=True, extra_cuda_cflags=['-fopenmp', '-ffast-math', '-O3', '-munsafe-fp-atomics'])
else:
    load(name="rwkv_mm8", sources=[f"{current_path}/cuda/wrapper.cpp", 
                                   f"{current_path}/cuda/operators.cu", 
                                   f"{current_path}/cuda/gemm_fp16_cublas.cpp"], is_python_module=False,
                    verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "--extra-device-vectorization",])

@MyStatic
def torch_mm8_seq(x, w, mx, rx, my, ry):
    return x @ ((w.to(dtype=x.dtype) + 0.5) * ry * rx + my + mx)

@MyStatic
def torch_mm8_one(x, w, mx, rx, my, ry):
    return x @ ((w.to(dtype=x.dtype) + 0.5) * ry * rx + my + mx)

@MyStatic
def cuda_mm8_seq(B: int, N: int, M: int, x, w, mx, rx, my, ry):
    y = torch.empty((B, M), device=w.device, dtype=x.dtype)
    torch.ops.rwkv.mm8_seq(B, N, M, x, w, mx, rx, my, ry, y)
    return y

@MyStatic
def cuda_mm8_one(N: int, M: int, x, w, mx, rx, my, ry):
    y = torch.zeros((M,), device=w.device, dtype=torch.float32)
    torch.ops.rwkv.mm8_one(N, M, x, w, mx, rx, my, ry, y)
    return y.to(dtype=x.dtype)

@MyStatic
def mm8_seq_cuda(x, w, mx, rx, my, ry):
    B, N, M = x.shape[0], w.shape[0], w.shape[1]
    return cuda_mm8_seq(B, N, M, x, w, mx, rx, my, ry)

@MyStatic
def mm8_one_cuda(x, w, mx, rx, my, ry):
    N, M = w.shape[0], w.shape[1]
    return cuda_mm8_one(N, M, x, w, mx, rx, my, ry)

@MyStatic
def mm8_cuda(x, w, mx, rx, my, ry):
    if len(x.shape) == 1:
        return mm8_one_cuda(x, w, mx, rx, my, ry)
    if len(x.shape) == 2:
        return mm8_seq_cuda(x, w, mx, rx, my, ry)
    # [B, T, C]
    B, T, C = x.shape
    y = mm8_seq_cuda(x.view(B * T, C), w, mx, rx, my, ry)
    return y.view(B, T, -1)

@MyStatic
def mm8_torch(x, w, mx, rx, my, ry):
    if len(x.shape) == 1:
        return torch_mm8_one(x, w, mx, rx, my, ry)
    if len(x.shape) == 2:
        return torch_mm8_seq(x, w, mx, rx, my, ry)
    # [B, T, C]
    B, T, C = x.shape
    y = torch_mm8_seq(x.view(B * T, C), w, mx, rx, my, ry)
    return y.view(B, T, -1)

@MyStatic
def linear_i8(x, w, mx, rx, my, ry):
    if w.dtype != torch.uint8:
        raise TypeError(f"linear_i8 only supports uint8 weight, got {w.dtype}")
    return mm8_cuda(x, w, mx, rx, my, ry)

@MyStatic
def linear_i8_torch(x, w, mx, rx, my, ry):
    if w.dtype != torch.uint8:
        raise TypeError(f"linear_i8 only supports uint8 weight, got {w.dtype}")
    return mm8_torch(x, w, mx, rx, my, ry)


@MyStatic
def linear_i8_bias(x, w, mx, rx, my, ry, bias):
    if w.dtype != torch.uint8:
        raise TypeError(f"linear_i8 only supports uint8 weight, got {w.dtype}")
    y = mm8_cuda(x, w, mx, rx, my, ry) + bias
    return y

@MyStatic
def linear_i8_bias_torch(x, w, mx, rx, my, ry, bias):
    if w.dtype != torch.uint8:
        raise TypeError(f"linear_i8 only supports uint8 weight, got {w.dtype}")
    y = mm8_torch(x, w, mx, rx, my, ry) + bias
    return y


def quantize_mm8_for_linear(w: torch.Tensor):
    # mm8 kernel expects weight as (N, M) where N is input dim.
    w_t = w.float().t().contiguous()
    if w_t.shape[0] > w_t.shape[1]:
        my = torch.amin(w_t, dim=1, keepdim=True)
        w_t = w_t - my
        mx = torch.amin(w_t, dim=0)
        w_t = w_t - mx
        rx = torch.amax(w_t, dim=0)
        rx = torch.clamp(rx, min=1e-8)
        w_t = w_t / rx
        ry = torch.amax(w_t, dim=1, keepdim=True)
        ry = torch.clamp(ry, min=1e-8)
        w_t = w_t / ry
    else:
        mx = torch.amin(w_t, dim=0)
        w_t = w_t - mx
        my = torch.amin(w_t, dim=1, keepdim=True)
        w_t = w_t - my
        rx = torch.amax(w_t, dim=0)
        rx = torch.clamp(rx, min=1e-8)
        w_t = w_t / rx
        ry = torch.amax(w_t, dim=1, keepdim=True)
        ry = torch.clamp(ry, min=1e-8)
        w_t = w_t / ry
    w_q = torch.clip(torch.floor(w_t * 256), min=0, max=255).to(dtype=torch.uint8)
    mx = mx.to(dtype=DTYPE).contiguous()
    rx = (rx / 16).to(dtype=DTYPE).contiguous()
    my = my.to(dtype=DTYPE).contiguous()
    ry = (ry / 16).to(dtype=DTYPE).contiguous()
    return w_q, mx, rx, my, ry
