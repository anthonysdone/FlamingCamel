# File

```python
"""
CUDA backend for FlamingCamel.
Handles GPU detection, PTX kernel loading, and kernel execution wrappers.
"""

import os
import cupy as cp
from pathlib import Path
from typing import Dict, Optional, Tuple


class CUDABackend:
    """Manages CUDA device and kernel operations."""
    
    def __init__(self):
        self._device_available = False
        self._kernels: Dict[str, cp.RawKernel] = {}
        self._device_id = 0
        
        # Initialize CUDA if available
        try:
            device_count = cp.cuda.runtime.getDeviceCount()
            if device_count > 0:
                self._device_available = True
                cp.cuda.Device(self._device_id).use()
                print(f"CUDA initialized: {device_count} device(s) found")
                
                # Print device info
                props = cp.cuda.runtime.getDeviceProperties(self._device_id)
                print(f"Using GPU: {props['name'].decode()}")
            else:
                print("No CUDA devices found, using CPU only")
        except Exception as e:
            print(f"CUDA initialization failed: {e}")
            print("Falling back to CPU only")
    
    @property
    def is_available(self) -> bool:
        """Check if CUDA is available."""
        return self._device_available
    
    def load_kernel(self, kernel_name: str, ptx_file: str) -> cp.RawKernel:
        """
        Load a CUDA kernel from PTX file.
        
        Args:
            kernel_name: Name of the kernel function (e.g., 'add_kernel')
            ptx_file: Path to PTX file relative to project root
        
        Returns:
            Compiled RawKernel ready to launch
        """
        # Cache check
        cache_key = f"{ptx_file}::{kernel_name}"
        if cache_key in self._kernels:
            return self._kernels[cache_key]
        
        # Find project root (where backend/ directory is)
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent  # frontend/ -> FlamingCamel/
        ptx_path = project_root / ptx_file
        
        if not ptx_path.exists():
            raise FileNotFoundError(
                f"PTX file not found: {ptx_path}\n"
                f"Make sure you've compiled kernels with: ./scripts/compile_kernels.sh"
            )
        
        # Load PTX code
        with open(ptx_path, 'r') as f:
            ptx_code = f.read()
        
        # Create kernel
        try:
            kernel = cp.RawKernel(ptx_code, kernel_name)
            self._kernels[cache_key] = kernel
            print(f"Loaded kernel: {kernel_name} from {ptx_file}")
            return kernel
        except Exception as e:
            raise RuntimeError(f"Failed to load kernel {kernel_name}: {e}")
    
    def get_grid_block_size(self, n: int, threads_per_block: int = 256) -> Tuple[int, int]:
        """
        Calculate grid and block dimensions for 1D kernel launch.
        
        Args:
            n: Number of elements to process
            threads_per_block: Threads per block (default 256)
        
        Returns:
            (grid_size, block_size) tuple
        """
        block_size = threads_per_block
        grid_size = (n + block_size - 1) // block_size
        return grid_size, block_size
    
    def check_cuda_error(self):
        """Check for CUDA errors and raise if any occurred."""
        if not self._device_available:
            return
        
        try:
            # Synchronize to catch any async errors
            cp.cuda.Stream.null.synchronize()
        except cp.cuda.runtime.CUDARuntimeError as e:
            raise RuntimeError(f"CUDA error: {e}")


# Global backend instance
_backend: Optional[CUDABackend] = None


def get_backend() -> CUDABackend:
    """Get the global CUDA backend instance (singleton)."""
    global _backend
    if _backend is None:
        _backend = CUDABackend()
    return _backend


def is_cuda_available() -> bool:
    """Check if CUDA is available."""
    return get_backend().is_available


# ============================================================================
# Kernel Wrapper Functions
# ============================================================================

def cuda_add(a: cp.ndarray, b: cp.ndarray, out: cp.ndarray) -> None:
    """
    Element-wise addition on GPU: out = a + b
    
    Args:
        a: Input array 1 (cupy array)
        b: Input array 2 (cupy array, same shape as a)
        out: Output array (cupy array, same shape as a)
    """
    backend = get_backend()
    
    if not backend.is_available:
        raise RuntimeError("CUDA not available")
    
    # Load kernel (cached after first call)
    kernel = backend.load_kernel('add_kernel', 'backend/ops_elementwise.ptx')
    
    # Get dimensions
    n = a.size
    grid_size, block_size = backend.get_grid_block_size(n)
    
    # Launch kernel
    # Signature: add_kernel(float* a, float* b, float* c, int n)
    kernel(
        (grid_size,),  # Grid dimensions
        (block_size,), # Block dimensions
        (a, b, out, n) # Kernel arguments
    )
    
    # Check for errors
    backend.check_cuda_error()


# ============================================================================
# Helper Functions
# ============================================================================

def synchronize():
    """Synchronize all CUDA operations (blocking)."""
    if is_cuda_available():
        cp.cuda.Stream.null.synchronize()


def empty_cache():
    """Clear the CUDA memory cache."""
    if is_cuda_available():
        cp.get_default_memory_pool().free_all_blocks()


def memory_info() -> Tuple[int, int]:
    """
    Get GPU memory info.
    
    Returns:
        (free_memory, total_memory) in bytes
    """
    if not is_cuda_available():
        return (0, 0)
    
    free, total = cp.cuda.runtime.memGetInfo()
    return (free, total)


def print_memory_info():
    """Print current GPU memory usage."""
    if not is_cuda_available():
        print("CUDA not available")
        return
    
    free, total = memory_info()
    used = total - free
    
    print(f"GPU Memory:")
    print(f"  Used:  {used / 1e9:.2f} GB")
    print(f"  Free:  {free / 1e9:.2f} GB")
    print(f"  Total: {total / 1e9:.2f} GB")
```


# Verify it works
```python
"""Test CUDA backend initialization and kernel loading."""

import numpy as np
import cupy as cp
from frontend.backend import get_backend, cuda_add, is_cuda_available


def test_backend_initialization():
    """Test backend initializes correctly."""
    backend = get_backend()
    print(f"CUDA available: {backend.is_available}")
    assert backend is not None


def test_cuda_add():
    """Test CUDA add kernel."""
    if not is_cuda_available():
        print("Skipping CUDA test: no GPU available")
        return
    
    # Create test data
    n = 1000
    a = cp.random.randn(n, dtype=cp.float32)
    b = cp.random.randn(n, dtype=cp.float32)
    out = cp.zeros(n, dtype=cp.float32)
    
    # Run CUDA kernel
    cuda_add(a, b, out)
    
    # Verify correctness
    expected = a + b
    np.testing.assert_allclose(
        cp.asnumpy(out),
        cp.asnumpy(expected),
        rtol=1e-5,
        atol=1e-6
    )
    
    print("✓ CUDA add kernel test passed")


def test_grid_block_calculation():
    """Test grid/block size calculation."""
    backend = get_backend()
    
    # Test various sizes
    test_cases = [
        (100, 256, 1, 256),      # Small: 1 block
        (256, 256, 1, 256),      # Exact fit
        (257, 256, 2, 256),      # Just over 1 block
        (1000, 256, 4, 256),     # Multiple blocks
        (10000, 256, 40, 256),   # Many blocks
    ]
    
    for n, threads, expected_grid, expected_block in test_cases:
        grid, block = backend.get_grid_block_size(n, threads)
        assert grid == expected_grid, f"n={n}: grid={grid}, expected={expected_grid}"
        assert block == expected_block, f"n={n}: block={block}, expected={expected_block}"
    
    print("✓ Grid/block calculation test passed")


if __name__ == "__main__":
    test_backend_initialization()
    test_grid_block_calculation()
    test_cuda_add()
```

# Elementwise ops .cu
```python
extern "C" __global__
void add_kernel(const float* a, const float* b, float* c, int n) {
    // Grid-stride loop pattern
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += stride) {
        c[i] = a[i] + b[i];
    }
}

// Placeholder for future kernels
extern "C" __global__
void mul_kernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += stride) {
        c[i] = a[i] * b[i];
    }
}

extern "C" __global__
void relu_kernel(const float* x, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += stride) {
        out[i] = fmaxf(x[i], 0.0f);
    }
}

extern "C" __global__
void relu_backward_kernel(const float* grad_out, const float* x, float* grad_in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += stride) {
        grad_in[i] = (x[i] > 0.0f) ? grad_out[i] : 0.0f;
    }
}
```

# Usage:
`./scripts/compile_kernels.sh`
`python tests/test_backend.py`
```python
from frontend.backend import cuda_add, is_cuda_available
import cupy as cp

if is_cuda_available():
    a = cp.array([1, 2, 3], dtype=cp.float32)
    b = cp.array([4, 5, 6], dtype=cp.float32)
    out = cp.zeros(3, dtype=cp.float32)
    
    cuda_add(a, b, out)
    print(out)  # [5, 7, 9]
```