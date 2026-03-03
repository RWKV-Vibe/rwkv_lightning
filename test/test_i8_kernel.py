from typing import List
import torch
import os
from torch.nn import functional as F
from torch.utils.cpp_extension import load

DTYPE = torch.half
ROCm_flag = torch.version.hip is not None
MyStatic = torch.jit.script 


if ROCm_flag == True:
    load(name="rwkv_mm8", sources=["infer/rwkv_batch/hip/wrapper.cpp", 
                                   "infer/rwkv_batch/hip/operators.hip", 
                                   "infer/rwkv_batch/hip/gemm_fp16_cublas.cpp"], is_python_module=False,
                    verbose=True, extra_cuda_cflags=['-fopenmp', '-ffast-math', '-O3', '-munsafe-fp-atomics'])
else:
    load(name="rwkv_mm8", sources=["infer/rwkv_batch/cuda/wrapper.cpp", 
                                   "infer/rwkv_batch/cuda/operators.cu", 
                                   "infer/rwkv_batch/cuda/gemm_fp16_cublas.cpp"], is_python_module=False,
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
    y = mm8_cuda(x, w, mx, rx, my, ry)
    return y

@MyStatic
def linear_i8_torch(x, w, mx, rx, my, ry):
    if w.dtype != torch.uint8:
        raise TypeError(f"linear_i8 only supports uint8 weight, got {w.dtype}")
    y = mm8_torch(x, w, mx, rx, my, ry)
    return y

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


def _quantize_mm8_for_linear(w: torch.Tensor):
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


def _max_abs_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def _run_mm8_linear_test(device: torch.device):
    x = torch.randn(64, device=device, dtype=DTYPE)
    w_ref = torch.randn(32, 64, device=device, dtype=DTYPE)
    bias = torch.randn(32, device=device, dtype=DTYPE)

    w_q, mx, rx, my, ry = _quantize_mm8_for_linear(w_ref)
    y_ref = F.linear(x, w_ref)
    y_out = linear_i8(
        x,
        w_q.to(device),
        mx.to(device),
        rx.to(device),
        my.to(device),
        ry.to(device)
    )
    y_out_torch = linear_i8_torch(
        x,
        w_q.to(device),
        mx.to(device),
        rx.to(device),
        my.to(device),
        ry.to(device)
    )

    err = _max_abs_error(y_ref, y_out)
    err_torch = _max_abs_error(y_ref, y_out_torch)
    print(f"[mm8 linear] max_abs_err={err:.6f}, max_abs_err_torch={err_torch:.6f}")


def _run_mm8_shape2_test(device: torch.device):
    # Explicitly exercise the len(x.shape) == 2 branch in mm8_cuda/mm8_torch.
    x = torch.randn(2, 2048, 2560, device=device, dtype=DTYPE)  # [batch_size, seq_len, hidden_dim]
    w_ref = torch.randn(10240, 2560, device=device, dtype=DTYPE)  # [hidden_dim, intermediate_dim] - FFN第一层
    bias = torch.randn(10240, device=device, dtype=DTYPE)  # [intermediate_dim] - 对应FFN第一层输出
    w_q, mx, rx, my, ry = _quantize_mm8_for_linear(w_ref)

    y_ref = F.linear(x, w_ref)
    y_out = linear_i8(
        x,
        w_q.to(device),
        mx.to(device),
        rx.to(device),
        my.to(device),
        ry.to(device)
    )
    y_out_torch = linear_i8_torch(
        x,
        w_q.to(device),
        mx.to(device),
        rx.to(device),
        my.to(device),
        ry.to(device)
    )
    print(f"[mm8 linear] y_ref.shape={y_ref.shape}, y_out.shape={y_out.shape}")

    err = _max_abs_error(y_ref, y_out)
    err_torch = _max_abs_error(y_ref, y_out_torch)
    print(f"[mm8 linear] max_abs_err={err:.6f}, max_abs_err_torch={err_torch:.6f}")

def _run_mm8_linear_test_with_bias(device: torch.device):
    x = torch.randn(64, device=device, dtype=DTYPE)
    w_ref = torch.randn(32, 64, device=device, dtype=DTYPE)
    bias = torch.randn(32, device=device, dtype=DTYPE)

    w_q, mx, rx, my, ry = _quantize_mm8_for_linear(w_ref)
    y_ref = F.linear(x, w_ref, bias)
    y_out = linear_i8_bias(
        x,
        w_q.to(device),
        mx.to(device),
        rx.to(device),
        my.to(device),
        ry.to(device),
        bias
    )
    y_out_torch = linear_i8_bias_torch(
        x,
        w_q.to(device),
        mx.to(device),
        rx.to(device),
        my.to(device),
        ry.to(device),
        bias
    )

    err = _max_abs_error(y_ref, y_out)
    err_torch = _max_abs_error(y_ref, y_out_torch)
    print(f"[mm8 linear] max_abs_err={err:.6f}, max_abs_err_torch={err_torch:.6f}")


def _run_mm8_shape2_test_with_bias(device: torch.device):
    # Explicitly exercise the len(x.shape) == 2 branch in mm8_cuda/mm8_torch.
    x = torch.randn(2, 2048, 2560, device=device, dtype=DTYPE)  # [batch_size, seq_len, hidden_dim]
    w_ref = torch.randn(10240, 2560, device=device, dtype=DTYPE)  # [hidden_dim, intermediate_dim] - FFN第一层
    bias = torch.randn(10240, device=device, dtype=DTYPE)  # [intermediate_dim] - 对应FFN第一层输出
    w_q, mx, rx, my, ry = _quantize_mm8_for_linear(w_ref)

    y_ref = F.linear(x, w_ref, bias)
    y_out = linear_i8_bias(
        x,
        w_q.to(device),
        mx.to(device),
        rx.to(device),
        my.to(device),
        ry.to(device),
        bias
    )
    y_out_torch = linear_i8_bias_torch(
        x,
        w_q.to(device),
        mx.to(device),
        rx.to(device),
        my.to(device),
        ry.to(device),
        bias
    )
    print(f"[mm8 linear] y_ref.shape={y_ref.shape}, y_out.shape={y_out.shape}")

    err = _max_abs_error(y_ref, y_out)
    err_torch = _max_abs_error(y_ref, y_out_torch)
    print(f"[mm8 linear] max_abs_err={err:.6f}, max_abs_err_torch={err_torch:.6f}")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("This test requires CUDA.")

    torch.manual_seed(42)
    device = torch.device("cuda")
    print(f"Running i8 kernel tests on {torch.cuda.get_device_name(0)}")
    print("======mm8 linear======")
    _run_mm8_linear_test(device)
    _run_mm8_shape2_test(device)
    print("======mm8 linear with bias======")
    _run_mm8_linear_test_with_bias(device)
    _run_mm8_shape2_test_with_bias(device)

    print("All i8 kernel tests passed.")


if __name__ == "__main__":
    main()
