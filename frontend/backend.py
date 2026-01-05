import os
import cupy as cp # type: ignore
from pathlib import Path
import ctypes

def cuda_add(a, b):
    backend = get_backend()
    if not backend.is_available:
        raise RuntimeError("CUDA is not available")
    
    if a.size != b.size: 
        raise ValueError(f"Size mismatch: {a.size} vs {b.size}")
    
    n = a.size
    grid_size, block_size = backend.get_grid_block_size(n)

    out = cp.empty_like(a)
    kernel = backend.load_kernel("add_kernel", "backend/ops_elementwise.cu")
    kernel(grid=(grid_size,), block=(block_size,), args=(a, b, out, n))

    return out

def cuda_add_backward(grad_out):
    backend = get_backend()
    if not backend.is_available:
        raise RuntimeError("CUDA is not available")
    
    n = grad_out.size
    grid_size, block_size = backend.get_grid_block_size(n)

    grad_a = cp.empty_like(grad_out)
    grad_b = cp.empty_like(grad_out)
    kernel = backend.load_kernel("add_backward_kernel", "backend/ops_elementwise.cu")
    kernel(grid=(grid_size,), block=(block_size,), args=(grad_out, grad_a, grad_b, n))

    return grad_a, grad_b

def cuda_mul(a, b): 
    backend = get_backend()
    if not backend.is_available:
        raise RuntimeError("CUDA is not available")
    
    if a.size != b.size: 
        raise ValueError(f"Size mismatch: {a.size} vs {b.size}")
    
    n = a.size
    grid_size, block_size = backend.get_grid_block_size(n)

    out = cp.empty_like(a)
    kernel = backend.load_kernel("mul_kernel", "backend/ops_elementwise.cu")
    kernel(grid=(grid_size,), block=(block_size,), args=(a, b, out, n))

    return out

def cuda_mul_backward(grad_out, a, b):
    backend = get_backend()
    if not backend.is_available:
        raise RuntimeError("CUDA is not available")
    
    n = grad_out.size
    grid_size, block_size = backend.get_grid_block_size(n)

    grad_a = cp.empty_like(grad_out)
    grad_b = cp.empty_like(grad_out)
    kernel = backend.load_kernel("mul_backward_kernel", "backend/ops_elementwise.cu")
    kernel(grid=(grid_size,), block=(block_size,), args=(grad_out, a, b, grad_a, grad_b, n))

    return grad_a, grad_b

def cuda_relu(x):
    backend = get_backend()
    if not backend.is_available:
        raise RuntimeError("CUDA is not available")

    n = x.size
    grid_size, block_size = backend.get_grid_block_size(n)
    out = cp.empty_like(x)
    kernel = backend.load_kernel("relu_kernel", "backend/ops_elementwise.cu")
    kernel(grid=(grid_size,), block=(block_size,), args=(x, out, n))

    return out

def cuda_relu_backward(grad_out, x):
    backend = get_backend()
    if not backend.is_available:
        raise RuntimeError("CUDA is not available")

    n = x.size
    grid_size, block_size = backend.get_grid_block_size(n)

    grad_x = cp.empty_like(grad_out)
    kernel = backend.load_kernel("relu_backward_kernel", "backend/ops_elementwise.cu")
    kernel(grid=(grid_size,), block=(block_size,), args=(grad_out, x, grad_x, n))

    return grad_x

class CUDABackend: 
    def __init__(self): 
        self.device_available = False
        self.kernels = {}
        self.device_id = 0

        try: 
            device_count = cp.cuda.runtime.getDeviceCount()
            if device_count > 0:
                self.device_available = True
                cp.cuda.Device(self.device_id).use()
                print(f"CUDA initialized: {device_count} device(s) found")

                props = cp.cuda.runtime.getDeviceProperties(self.device_id)
                print(f"Using GPU: {props['name'].decode()}")
            else: 
                print("No CUDA devices found, using CPU only")
        except Exception as e: 
            print(f"CUDA initialization failed: {e}")
            print("Falling back to CPU only")

    @property
    def is_available(self): 
        return self.device_available
    
    def load_kernel(self, kernel_name, cuda_file): 
        cache_key = f"{cuda_file}::{kernel_name}"
        if cache_key in self.kernels: 
            return self.kernels[cache_key]

        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent
        cuda_path = project_root / cuda_file

        if not cuda_path.exists():
            raise FileNotFoundError(f"CUDA source file not found: {cuda_path}\n")
        
        with open(cuda_path, "r") as f:
            cuda_code = f.read()
        
        try: 
            kernel = cp.RawKernel(cuda_code, kernel_name)
            self.kernels[cache_key] = kernel
            print(f"Loaded kernel: {kernel_name} from {cuda_file}")
            return kernel
        except Exception as e: 
            raise RuntimeError(f"Failed to load kernel {kernel_name}: {e}")
        
    def get_grid_block_size(self, n, threads_per_block=256): 
        block_size = threads_per_block
        grid_size = (n + block_size - 1) // block_size
        return grid_size, block_size

    def check_cuda_error(self): 
        if not self.device_available:
            return 
        
        try: 
            cp.cuda.Stream.null.synchronize()
        except cp.cuda.runtime.CUDARuntimeError as e: 
            raise RuntimeError(f"CUDA error: {e}")
        
backend = None

def get_backend():
    global backend
    if backend is None: 
        backend = CUDABackend()
    return backend

def is_cuda_available(): 
    return get_backend().is_available

def synchronize():
    if is_cuda_available():
        cp.cuda.Stream.null.synchronize()

def empty_cache():
    if is_cuda_available():
        cp.get_default_memory_pool().free_all_blocks()
    
def memory_info():
    if not is_cuda_available():
        return (0, 0)
    
    free, total = cp.cuda.runtime.memGetInfo()
    return (free, total)

def print_memory_info():
    if not is_cuda_available():
        print("CUDA not available")
        return

    free, total = memory_info()
    used = total - free

    print(f"GPU Memory:")
    print(f"  Used:  {used / 1e9:.2f} GB")
    print(f"  Free:  {free / 1e9:.2f} GB")
    print(f"  Total: {total / 1e9:.2f} GB")