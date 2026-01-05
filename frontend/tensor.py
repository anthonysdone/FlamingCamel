import numpy as np
try:
    import cupy as cp
    CUDA = True
except ImportError: 
    CUDA = False

from frontend.autograd import *

class Tensor: 
    def __init__(self, data, requires_grad=False, device="cuda"): 
        if isinstance(data, np.ndarray):
            self.data = data
        elif isinstance(data, cp.ndarray):
            self.data = data
        else:
            if CUDA and device == "cuda":
                self.data = cp.array(data, dtype=cp.float32)
            else:
                self.data = np.array(data, dtype=np.float32)

        self.requires_grad = requires_grad
        self.grad = None

        self.ctx = None
        self.backward_fn = None
        self.prev_tensors = []
    
    def backward(self): 
        if self.data.shape != (): 
            raise RuntimeError("backward() is only defined on scalar tensors")
        backward_pass(self)
    
    def item(self): 
        return self.data.item()
    
    def numpy(self): 
        if CUDA and isinstance(self.data, cp.ndarray):
            return cp.asnumpy(self.data)
        return self.data
    
    def cupy(self): 
        if CUDA and isinstance(self.data, np.ndarray):
            return cp.array(self.data)
        elif isinstance(self.data, cp.ndarray):
            return self.data
        else:
            raise RuntimeError("CUDA is not available")

    @property
    def shape(self): 
        return self.data.shape
    
    @property
    def dtype(self):
        return self.data.dtype
    
    def device(self):
        if CUDA and isinstance(self.data, cp.ndarray):
            return "cuda"
        else: 
            return "cpu"
    
    def to(self, device):
        if device == "cuda":
            if not CUDA:
                raise RuntimeError("CUDA not available.")
            if isinstance(self.data, cp.ndarray):
                return self
            self.data = cp.array(self.data)
            return self

        elif device == "cpu":
            if isinstance(self.data, np.ndarray):
                return self
            if CUDA and isinstance(self.data, cp.ndarray):
                self.data = cp.asnumpy(self.data)
            self.data = np.array(self.data)
            return self
        else:
            raise ValueError(f"Unknown device: {device}")
    
    def __add__(self, other): 
        from .functional import add
        if not isinstance(other, Tensor):
            other = Tensor(other, device=self.device())
        return add(self, other)
    
    def __mul__(self, other): 
        from .functional import mul
        if not isinstance(other, Tensor):
            other = Tensor(other, device=self.device())
        return mul(self, other)
    
    def __matmul__(self, other): 
        from .functional import matmul
        return matmul(self, other)
    
    def relu(self): 
        from .functional import relu
        return relu(self)
    
    def __repr__(self): 
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"
    

def tensor(data, requires_grad=False, device="cuda"): 
    return Tensor(data, requires_grad, device)

def zeros(*shape, requires_grad=False, device="cuda"): 
    if device == "cuda" and CUDA:
        return Tensor(cp.zeros(shape, dtype=cp.float32), requires_grad, device)
    return Tensor(np.zeros(shape, dtype=np.float32), requires_grad, device="cpu")

def ones(*shape, requires_grad=False, device="cuda"): 
    if device == "cuda" and CUDA:
        return Tensor(cp.ones(shape, dtype=cp.float32), requires_grad, device)
    return Tensor(np.ones(shape, dtype=np.float32), requires_grad, device="cpu")

def randn(*shape, requires_grad=False, device="cuda"): 
    if device == "cuda" and CUDA:
        return Tensor(cp.random.randn(*shape).astype(cp.float32), requires_grad, device)
    return Tensor(np.random.randn(*shape).astype(np.float32), requires_grad, device="cpu")

def arrange(start, stop=None, step=1, requires_grad=False, device="cuda"):
    if stop is None: 
        stop = start
        start = 0
    if device == "cuda" and CUDA:
        return Tensor(cp.arange(start, stop, step, dtype=cp.float32), requires_grad, device)
    return Tensor(np.arange(start, stop, step, dtype=np.float32), requires_grad, device="cpu")

class Parameter(Tensor):
    def __init__(self, data, device="cuda"): 
        super().__init__(data, True, device)