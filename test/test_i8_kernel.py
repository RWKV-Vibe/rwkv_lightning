import sys
from pathlib import Path
import time
import torch
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from infer.rwkv_batch.rwkv7.ops.mm_int8_kernel import *
from convert_int8_weight import quantize_w8a8_weight_cpu

DTYPE = torch.half
ROCm_flag = torch.version.hip is not None
MyStatic = torch.jit.script 

def _error_stats(ref: torch.Tensor, out: torch.Tensor) -> dict[str, float]:
    ref_f = ref.float()
    out_f = out.float()
    diff = (ref_f - out_f).abs()
    rel = diff / ref_f.abs().clamp_min(1e-6)
    rmse = torch.sqrt(torch.mean((ref_f - out_f) ** 2))
    ref_rms = torch.sqrt(torch.mean(ref_f ** 2)).clamp_min(1e-6)
    return {
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "rmse": rmse.item(),
        "nrmse": (rmse / ref_rms).item(),
        "max_rel": rel.max().item(),
        "mean_rel": rel.mean().item(),
        "ref_max": ref_f.abs().max().item(),
        "ref_mean": ref_f.abs().mean().item(),
        "ref_rms": ref_rms.item(),
    }


def _format_error_stats(stats: dict[str, float]) -> str:
    return (
        f"max_abs={stats['max_abs']:.6f}, "
        f"mean_abs={stats['mean_abs']:.6f}, "
        f"rmse={stats['rmse']:.6f}, "
        f"nrmse={stats['nrmse']:.6f}, "
        f"max_rel={stats['max_rel']:.6f}, "
        f"mean_rel={stats['mean_rel']:.6f}, "
        f"ref_max={stats['ref_max']:.6f}, "
        f"ref_mean={stats['ref_mean']:.6f}, "
        f"ref_rms={stats['ref_rms']:.6f}"
    )


def _assert_w8a8_total_error(stats: dict[str, float]):
    # W8A8 compares against original FP16 linear, so this includes both weight
    # quantization and activation quantization error. The relative max can blow
    # up near zero, so assert on absolute and aggregate errors instead.
    assert stats["max_abs"] < 0.30, f"W8A8 max_abs too high: {stats['max_abs']:.6f}"
    assert stats["rmse"] < 0.10, f"W8A8 rmse too high: {stats['rmse']:.6f}"


def _run_mm8_linear_test(device: torch.device):
    x = torch.randn(64, device=device, dtype=DTYPE)
    w_ref = torch.randn(32, 64, device=device, dtype=DTYPE)

    w_q, mx, rx, my, ry = quantize_mm8_for_linear(w_ref)
    y_ref = F.linear(x, w_ref)
    y_out = linear_i8(
        x,
        w_q.to(device),
        mx.to(device),
        rx.to(device),
        my.to(device),
        ry.to(device)
    )

    stats = _error_stats(y_ref, y_out)
    _assert_w8a8_total_error(stats)
    print(f"[mm8 linear W8A8 vs FP16] {_format_error_stats(stats)}")

def _run_mm8_linear_test_with_bias(device: torch.device):
    x = torch.randn(64, device=device, dtype=DTYPE)
    w_ref = torch.randn(32, 64, device=device, dtype=DTYPE)
    bias = torch.randn(32, device=device, dtype=DTYPE)

    w_q, mx, rx, my, ry = quantize_mm8_for_linear(w_ref)
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

    stats = _error_stats(y_ref, y_out)
    _assert_w8a8_total_error(stats)
    print(f"[mm8 linear bias W8A8 vs FP16] {_format_error_stats(stats)}")


def _run_prequant_linear_w8a8_test(device: torch.device):
    x = torch.randn(37, 64, device=device, dtype=DTYPE)
    w_ref = torch.randn(32, 64, device=device, dtype=DTYPE)

    w_q, scale = quantize_w8a8_weight_cpu(w_ref)
    y_ref = F.linear(x, w_ref)
    y_out = linear_w8a8(x, w_q.to(device), scale.to(device))

    stats = _error_stats(y_ref, y_out)
    _assert_w8a8_total_error(stats)
    print(f"[prequant linear_w8a8 vs FP16] {_format_error_stats(stats)}")


def _run_prequant_linear_w8a8_3d_test(device: torch.device):
    x = torch.randn(5, 7, 64, device=device, dtype=DTYPE)
    w_ref = torch.randn(32, 64, device=device, dtype=DTYPE)

    w_q, scale = quantize_w8a8_weight_cpu(w_ref)
    y_ref = F.linear(x, w_ref)
    y_out = linear_w8a8(x, w_q.to(device), scale.to(device))

    stats = _error_stats(y_ref, y_out)
    _assert_w8a8_total_error(stats)
    print(f"[prequant linear_w8a8 3D vs FP16] {_format_error_stats(stats)}")

def _benchmark_mm8_linear_large_batch(device: torch.device):
    print("======mm8 linear large-batch benchmark======")
    # 固定 weight shape: [out_features=4096, in_features=4096]
    out_features, in_features = 4096, 4096
    w_ref = torch.randn(out_features, in_features, device=device, dtype=DTYPE)
    w_q, mx, rx, my, ry = quantize_mm8_for_linear(w_ref)

    # 测试多个 batch sizes
    batch_sizes = [1, 8, 32, 128, 512, 1024]
    warmup = 5
    repeats = 10

    for B in batch_sizes:
        x = torch.randn(B, in_features, device=device, dtype=DTYPE)

        # Warmup
        for _ in range(warmup):
            _ = linear_i8(x, w_q, mx, rx, my, ry)
        torch.cuda.synchronize()

        # Benchmark i8 kernel
        start = time.time()
        for _ in range(repeats):
            y_i8 = linear_i8(x, w_q, mx, rx, my, ry)
        torch.cuda.synchronize()
        i8_time = (time.time() - start) / repeats

        # Benchmark reference FP16
        for _ in range(warmup):
            _ = F.linear(x, w_ref)
        torch.cuda.synchronize()

        start = time.time()
        for _ in range(repeats):
            y_fp16 = F.linear(x, w_ref)
        torch.cuda.synchronize()
        fp16_time = (time.time() - start) / repeats

        # Compute throughput (GFLOPs)
        # GEMM FLOPs = 2 * B * M * N
        flops = 2 * B * out_features * in_features
        gflops_i8 = flops / i8_time / 1e9
        gflops_fp16 = flops / fp16_time / 1e9
        stats = _error_stats(y_fp16, y_i8)

        print(f"Batch={B:4d} | i8: {i8_time*1000:.2f} ms ({gflops_i8:.1f} GFLOPs/s) "
              f"| FP16: {fp16_time*1000:.2f} ms ({gflops_fp16:.1f} GFLOPs/s) "
              f"| Speedup: {gflops_i8/gflops_fp16:.2f}x "
              f"| err: max_abs={stats['max_abs']:.4f}, rmse={stats['rmse']:.4f}, "
              f"nrmse={stats['nrmse']:.4f}, mean_rel={stats['mean_rel']:.4f}")
def main():
    if not torch.cuda.is_available():
        raise RuntimeError("This test requires CUDA.")

    torch.manual_seed(42)
    device = torch.device("cuda")
    print(f"Running i8 kernel tests on {torch.cuda.get_device_name(0)}")
    print("======mm8 linear======")
    _run_mm8_linear_test(device)
    print("======mm8 linear with bias======")
    _run_mm8_linear_test_with_bias(device)
    print("======prequant linear_w8a8======")
    _run_prequant_linear_w8a8_test(device)
    print("======prequant linear_w8a8 3D======")
    _run_prequant_linear_w8a8_3d_test(device)

    _benchmark_mm8_linear_large_batch(device)

    print("All i8 kernel tests passed.")


if __name__ == "__main__":
    main()
