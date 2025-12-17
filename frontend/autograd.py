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

    tensor.grad = tensor.data.__class__(1.0) if tensor.data.shape == () else tensor.data.__class__.ones(tensor.data.shape)

    for t in reversed(topo): 
        grads = t.backward_fn(t.ctx, t.grad)
        if not isinstance(grads, tuple): 
            grads = (grads,)
        
        for prev_tensor, grad in zip(t.prev_tensors, grads): 
            if grad is not None: 
                if prev_tensor.grad is None: 
                    prev_tensor.grad = grad
                else: 
                    prev_tensor.grad += grad