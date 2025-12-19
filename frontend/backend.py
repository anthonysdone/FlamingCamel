import os
import cupy as cp
from pathlib import Path

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