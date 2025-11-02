import torch
import time
import sys
import os

current_path = os.path.dirname(__file__)
sys.path.insert(0, current_path)

from rwkv_mm_op_triton import rwkv_mm_sparsity


def test_performance():
    print("\n" + "="*60)
    print("测试 4: 性能测试")
    print("="*60)
    
    device = torch.device("cuda")
    
    test_sizes = [
        (1024, 512),
        (4096, 2048),
        (16384, 4096),
        (32768, 8192),
    ]
    
    num_iterations = 100
    num_warmup = 50
    
    for K, C in test_sizes:
        print(f"\n测试尺寸: K={K}, C={C}")
        
        k = torch.randn(K, device=device, dtype=torch.half)
        k[torch.randperm(K)[:K//2]] = 0.0
        v = torch.randn(K, C, device=device, dtype=torch.half)
        
        # Warm up
        for _ in range(num_warmup):
            _ = rwkv_mm_sparsity(k, v)
            _ = k @ v
        torch.cuda.synchronize()
        
        start_time = time.time()
        for _ in range(num_iterations):
            triton_out = rwkv_mm_sparsity(k, v)
        torch.cuda.synchronize()
        triton_time = time.time() - start_time
        
        start_time = time.time()
        for _ in range(num_iterations):
            torch_out = k @ v
        torch.cuda.synchronize()
        torch_time = time.time() - start_time
        
        print(f"  Triton 实现: {triton_time/num_iterations*1000:.4f} ms/iter")
        print(f"  PyTorch 实现: {torch_time/num_iterations*1000:.4f} ms/iter")
        
        if torch_time > 0:
            speedup = torch_time / triton_time
            print(f"  加速比: {speedup:.2f}x")
            if speedup > 1:
                print(f"  ✓ Triton 实现快 {speedup:.2f}x")
            else:
                print(f"  ⚠️  PyTorch 实现快 {1/speedup:.2f}x")


def main():
    print("\n" + "="*60)
    print("RWKV MM Sparsity Triton 算子测试")
    print("="*60)
    
    device = torch.cuda.get_device_name(0)
    print(f"\n使用设备: {device}")
    print(f"CUDA 版本: {torch.version.cuda}")
    
    test_performance()
    
    print("\n" + "="*60)
    print("所有测试完成！")
    print("="*60)


if __name__ == "__main__":
    main()

