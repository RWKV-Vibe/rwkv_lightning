import os
from torch.utils.cpp_extension import load
import torch
import time

current_path = os.path.dirname(__file__)
sources = [
    os.path.join(current_path, "hip/rwkv_mm_sparsity_op.hip"),
    os.path.join(current_path, "hip/rwkv_mm_sparsity.hip"),
]

HEAD_SIZE = 64  

extra_cflags = [
    "-fopenmp",
    "-ffast-math",
    "-O3",
    f"--offload-arch=gfx1030",
    "-munsafe-fp-atomics",
    f"-D_N_={HEAD_SIZE}",
    "-x", "hip"
]

# sometimes you need to pass include paths or link flags
extra_include_paths = [ os.path.join(current_path, "hip") ]

print("Compiling RWKV HIP extension...")

load(
    name="rwkv_mm_sparsity_hip",
    sources=sources,
    verbose=True,
    extra_cflags=extra_cflags,
    extra_include_paths=extra_include_paths,
    is_python_module=False
)

if torch.cuda.is_available() or torch.version.hip is not None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    K = 16384
    C = 4096
    k = torch.randn(K, device=device, dtype=torch.float32)
    # sparsify k
    k[torch.randperm(K)[:K//2]] = 0.0
    v = torch.randn(K, C, device=device, dtype=torch.float32)

    # Warm up GPU
    for _ in range(50):
        _ = torch.ops.rwkv_mm_sparsity_hip.rwkv_mm_sparsity(k, v)
        _ = k @ v
    torch.cuda.synchronize()

    # 测试自定义HIP内核性能
    num_iterations = 100
    start_time = time.time()
    for i in range(num_iterations):
        out = torch.ops.rwkv_mm_sparsity_hip.rwkv_mm_sparsity(k, v)
    torch.cuda.synchronize()
    hip_time = time.time() - start_time

    # 测试PyTorch原生矩阵乘法性能
    start_time = time.time()
    for i in range(num_iterations):
        torch_result = k @ v
    torch.cuda.synchronize()
    torch_time = time.time() - start_time

    # 输出结果和性能对比
    print("hip kernel out shape:", out.shape)
    print("torch result shape:", torch_result.shape)
    
    diff = torch.abs(out - torch_result)
    max_diff = torch.max(diff)
    mean_diff = torch.mean(diff)
    
    print("\n=== 精度对比 ===")
    print("Maximum absolute difference:", max_diff.item())
    print("Mean absolute difference:", mean_diff.item())
    print("Results match:", torch.allclose(out, torch_result, atol=1e-6))
    
    print("\n=== 性能对比 ===")
    print(f"HIP kernel time ({num_iterations} iterations): {hip_time:.4f}s")
    print(f"PyTorch time ({num_iterations} iterations): {torch_time:.4f}s")
    print(f"Average HIP kernel time per iteration: {hip_time/num_iterations*1000:.4f}ms")
    print(f"Average PyTorch time per iteration: {torch_time/num_iterations*1000:.4f}ms")
    
    if torch_time > 0:
        speedup = torch_time / hip_time
        print(f"Speedup ratio (PyTorch/HIP): {speedup:.2f}x")
        if speedup > 1:
            print(f"HIP kernel is {speedup:.2f}x faster than PyTorch")
        else:
            print(f"PyTorch is {1/speedup:.2f}x faster than HIP kernel")
    
else:
    print("No ROCm device available in this environment.")