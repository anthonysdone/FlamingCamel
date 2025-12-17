import numpy as np
from .autograd import backward_pass

class Tensor: 
    def __init__(self, data, requires_grad=False): 
        if not isinstance(data, np.ndarray): 
            data = np.array(data, dtype=np.float32)
        self.data = data

        self.requires_grad = requires_grad
        self.grad = None

        self.ctx = None
        self.backward_fn = None
        self.prev_tensors = []
    
    def backward(self): 
        if self.data.shape != (): 
            raise RuntimeError
        backward_pass(self)
    
    def item(self): 
        return self.data.item()
    
    def numpy(self): 
        return self.data
    
    @property
    def shape(self): 
        return self.data.shape
    
    @property
    def dtype(self):
        return self.data.dtype
    
    def __add__(self, other): 
        from .functional import add
        return add(self, other)
    
    def __mul__(self, other): 
        from .functional import mul
        return mul(self, other)
    
    def __matmul__(self, other): 
        from .functional import matmul
        return matmul(self, other)
    
    def relu(self): 
        from .functional import relu
        return relu(self)
    
    def __repr__(self): 
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"
    
def tensor(data, requires_grad=False): 
    return Tensor(data, requires_grad)

def zeros(*shape, requires_grad=False): 
    return Tensor(np.zeros(shape, dtype=np.float32), requires_grad)

def ones(*shape, requires_grad=False): 
    return Tensor(np.ones(shape, dtype=np.float32), requires_grad)

def randn(*shape, requires_grad=False): 
    return Tensor(np.random.randn(*shape).astype(np.float32), requires_grad)

def arrange(start, stop=None, step=1, requires_grad=False):
    if stop is None: 
        stop = start
        start = 0
    return Tensor(np.arange(start, stop, step, dtype=np.float32), requires_grad)

class Parameter(Tensor):
    def __init__(self, data): 
        super().__init__(data, True)