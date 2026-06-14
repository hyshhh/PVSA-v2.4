"""Benchmark CUDA kernel performance comparison."""
import torch
import time
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mmseg.models.utils.topp_flash_kernel import (
    topp_flash_attention,
    _topp_attention_block_impl,
    _load_cuda_extension,
    is_topp_flash_available,
)


def create_test_data(batch_size=1, n_win=7, q_len=16, kv_len=16, dim=512, qk_dim=256, num_heads=8, topk=12, device='cuda'):
    """Create test tensors for benchmarking."""
    p2 = n_win * n_win
    kv_dim = qk_dim + dim
    
    q_pix = torch.randn(batch_size, p2, q_len, qk_dim, device=device, dtype=torch.float32)
    kv_pix = torch.randn(batch_size, p2, kv_len, kv_dim, device=device, dtype=torch.float32)
    
    # Create routing indices
    r_idx = torch.zeros(batch_size, p2, topk, device=device, dtype=torch.long)
    for i in range(p2):
        available = list(range(p2))
        available.remove(i)
        selected = torch.tensor(available[:topk-1] + [i], device=device)
        r_idx[:, i, :] = selected
    
    r_weight = torch.softmax(torch.randn(batch_size, p2, topk, device=device), dim=-1)
    r_mask = torch.ones(batch_size, p2, topk, device=device, dtype=torch.bool)
    
    return q_pix, kv_pix, r_weight, r_idx, r_mask


def benchmark_function(func, args, warmup=10, iterations=100, name=""):
    """Benchmark a function."""
    # Warmup
    for _ in range(warmup):
        _ = func(*args)
    torch.cuda.synchronize()
    
    # Benchmark
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    
    for i in range(iterations):
        start_events[i].record()
        _ = func(*args)
        end_events[i].record()
    
    torch.cuda.synchronize()
    
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"{name}:")
    print(f"  Average: {avg_time:.3f} ms")
    print(f"  Min: {min_time:.3f} ms")
    print(f"  Max: {max_time:.3f} ms")
    print(f"  FPS: {1000.0 / avg_time:.2f}")
    
    return avg_time


def main():
    print("=" * 60)
    print("CUDA Kernel Performance Benchmark")
    print("=" * 60)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("CUDA is not available!")
        return
    
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print()
    
    # Create test data
    print("Creating test data...")
    q_pix, kv_pix, r_weight, r_idx, r_mask = create_test_data(
        batch_size=1, n_win=7, q_len=16, kv_len=16, dim=512, qk_dim=256, num_heads=8, topk=12
    )
    
    H = 7 * 4  # n_win * sqrt(q_len)
    W = 7 * 4
    
    # Benchmark torch_block backend
    print("\n1. torch_block backend:")
    try:
        time_torch_block = benchmark_function(
            _topp_attention_block_impl,
            (q_pix, kv_pix, r_weight, r_idx, r_mask, 8, 256, 512, 256**-0.5, 7, H, W, 64),
            name="torch_block"
        )
    except Exception as e:
        print(f"  Error: {e}")
        time_torch_block = float('inf')
    
    # Benchmark CUDA kernel
    print("\n2. CUDA kernel:")
    if is_topp_flash_available('cuda'):
        try:
            extension = _load_cuda_extension()
            keep_len = r_mask.sum(dim=-1).contiguous().long()
            
            time_cuda = benchmark_function(
                extension.forward,
                (q_pix.contiguous(), kv_pix.contiguous(), r_weight.contiguous(), 
                 r_idx.contiguous().long(), keep_len, 8, 256, 512, 256**-0.5, 7, H, W),
                name="cuda_kernel"
            )
        except Exception as e:
            print(f"  Error: {e}")
            time_cuda = float('inf')
    else:
        print("  CUDA kernel not available")
        time_cuda = float('inf')
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  torch_block: {time_torch_block:.3f} ms")
    print(f"  cuda_kernel: {time_cuda:.3f} ms")
    
    if time_torch_block < float('inf') and time_cuda < float('inf'):
        speedup = time_torch_block / time_cuda
        print(f"  Speedup: {speedup:.2f}x")
        
        if speedup > 1:
            print("  CUDA kernel is faster!")
        else:
            print("  torch_block is faster")
    
    print("=" * 60)


if __name__ == '__main__':
    main()
