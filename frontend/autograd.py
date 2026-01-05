import numpy as np

class Context: 
    def __init__(self): 
        self.saved_tensors = []
        self.saved_data = {}

    def save_for_backward(self, *tensors): 
        self.saved_tensors.extend(tensors)
    
    def save(self, **kwargs): 
        self.saved_data.update(kwargs)
    
class Function: 
    @staticmethod
    def forward(ctx, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output): 
        raise NotImplementedError
    
    @classmethod
    def apply(cls, *args, **kwargs): 
        ctx = Context()
        output = cls.forward(ctx, *args, **kwargs) 

        if any(getattr(arg, "requires_grad", False) for arg in args if hasattr(arg, "requires_grad")): 
            output.ctx = ctx
            output.backward_fn = cls.backward
            output.prev_tensors = [arg for arg in args if hasattr(arg, "requires_grad")]

        return output
    
def backward_pass(tensor): 
    topo = []
    visited = set()

    def build_topo(t): 
        if t not in visited and hasattr(t, "backward_fn"): 
            visited.add(t) 
            for prev in t.prev_tensors: 
                build_topo(prev)
            topo.append(t)
    
    build_topo(tensor)

    # Initialize gradient
    try:
        import cupy as cp
        CUDA = True
    except ImportError:
        CUDA = False
    
    if tensor.data.shape == ():
        if CUDA and isinstance(tensor.data, cp.ndarray):
            # CuPy array
            tensor.grad = cp.array(1.0)
        else:
            tensor.grad = np.array(1.0)
    else:
        if CUDA and isinstance(tensor.data, cp.ndarray):
            # CuPy array
            tensor.grad = cp.ones(tensor.data.shape)
        else:
            tensor.grad = np.ones(tensor.data.shape)

    for t in reversed(topo): 
        if t.backward_fn is None:
            continue
            
        grads = t.backward_fn(t.ctx, t.grad)
        if not isinstance(grads, tuple): 
            grads = (grads,)
        
        for prev_tensor, grad in zip(t.prev_tensors, grads): 
            if grad is not None: 
                # Ensure gradient is on same device as tensor
                if CUDA and isinstance(prev_tensor.data, cp.ndarray) and isinstance(grad, cp.ndarray):
                    # Both on GPU
                    pass
                elif CUDA and isinstance(prev_tensor.data, cp.ndarray) and isinstance(grad, np.ndarray):
                    # Tensor on GPU, grad on CPU - convert grad to GPU
                    grad = cp.array(grad)
                elif CUDA and isinstance(grad, cp.ndarray) and isinstance(prev_tensor.data, np.ndarray):
                    # Tensor on CPU, grad on GPU - convert grad to CPU
                    grad = cp.asnumpy(grad)
                
                if prev_tensor.grad is None: 
                    prev_tensor.grad = grad
                else: 
                    prev_tensor.grad = prev_tensor.grad + grad