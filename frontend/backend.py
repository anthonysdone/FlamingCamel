import os
import cupy as cp # type: ignore
from pathlib import Path


def cuda_add(a, b, out): 
    backend = get_backend()

    if not backend.is_available: 
        raise RuntimeError("CUDA not available")

    kernel = backend.load_kernel("add_kernel", "backend/ops_elementwise.ptx")

    n = a.size
    grid_size, block_size = backend.get_grid_block_size(n)

    kernel(
        (grid_size,),
        (block_size,),
        (a, b, out, n)
    )

    backend.check_cuda_error()


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
                print(f"Using GPU: {props["name"].decode()}")
            else: 
                print("No CUDA devices found, using CPU only")
        except Exception as e: 
            print(f"CUDA initialization failed: {e}")
            print("Falling back to CPU only")

    @property
    def is_available(self): 
        return self.device_available
    
    def load_kernel(self, kernel_name, ptx_file): 
        cache_key = f"{ptx_file}::{kernel_name}"
        if cache_key in self.kernels: 
            return self.kernels[cache_key]

        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent
        ptx_path = project_root / ptx_file

        if not ptx_path.exists():
            raise FileNotFoundError(
                f"PTX file not found: {ptx_path}\n"
                f"Make sure you have compiled kernels with: ./scripts/compile_kernels.sh"
            )
        
        with open(ptx_path, "r") as f:
            ptx_code = f.read()
        
        try: 
            kernel = cp.RawKernel(ptx_code, kernel_name)
            self.kernels[cache_key] = kernel
            print(f"Loaded kernel: {kernel_name} from {ptx_file}")
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
        cp.cuda.Strem.null.synchronize()

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