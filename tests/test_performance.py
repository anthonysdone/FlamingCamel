import numpy as np
import time 
import sys
import os

try: 
    import cupy as cp
    CUDA = True
except ImportError:
    CUDA = False

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tensor import Tensor
from functional import add, mul, matmul, relu

# pip install cupy-cuda12x 

def benchmark_op(name, op_fn, *args, iterations=100, device="cpu"):
    # Warmup
    for _ in range(5):
        op_fn(*args)
    
    if device == "cuda":
        cp.cuda.runtime.deviceSynchronize()
    
    start = time.perf_counter()
    for _ in range(iterations):
        op_fn(*args)
    
    if device == "cuda":
        cp.cuda.runtime.deviceSynchronize()
    end = time.perf_counter()

    avg_time_ms = (end - start) / iterations * 1000
    return avg_time_ms

def benchmark_ewise_ops():
    print("\nBenchmarking elementwise operations...")
    print("=" * 30)

    sizes = [100000, 1000000, 10000000]

    for size  in sizes: 
        print(f"Size: {size:,} elements")
        
        a_cpu = Tensor(np.random.randn(size), requires_grad=True, device="cpu")
        b_cpu = Tensor(np.random.randn(size), requires_grad=True, device="cpu")

        add_cpu = benchmark_op("add_cpu", lambda: add(a_cpu, b_cpu), device="cpu")
        mul_cpu = benchmark_op("mul_cpu", lambda: mul(a_cpu, b_cpu), device="cpu")
        relu_cpu = benchmark_op("relu_cpu", lambda: relu(a_cpu), device="cpu")

        print(f"  CPU:      add={add_cpu:.4f}ms  mul={mul_cpu:.4f}ms  relu={relu_cpu:.4f}ms")

        if CUDA:
            a_cuda = Tensor(cp.random.randn(size), requires_grad=True, device="cuda")
            b_cuda = Tensor(cp.random.randn(size), requires_grad=True, device="cuda")

            add_cuda = benchmark_op("add_cuda", lambda: add(a_cuda, b_cuda), device="cuda")
            mul_cuda = benchmark_op("mul_cuda", lambda: mul(a_cuda, b_cuda), device="cuda")
            relu_cuda = benchmark_op("relu_cuda", lambda: relu(a_cuda), device="cuda")

            print(f"  CUDA:     add={add_cuda:.4f}ms  mul={mul_cuda:.4f}ms  relu={relu_cuda:.4f}ms")
            print(f"  Speedup:  add={add_cpu/add_cuda:.2f}x  mul={mul_cpu/mul_cuda:.2f}x  relu={relu_cpu/relu_cuda:.2f}x")
        
    print("=" * 30)
    print("Elementwise operations benchmark completed!")

if __name__ == "__main__":
    benchmark_ewise_ops()