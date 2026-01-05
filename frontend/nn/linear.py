import numpy as np
from ..tensor import Tensor, Parameter
from .module import Module

try:
    import cupy as cp
    CUDA = True
except ImportError:
    CUDA = False

class Linear(Module): 
    def __init__(self, in_features, out_features, bias=True): 
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # kaiming he is at mit apparently woah
        k = 1.0 / in_features
        bound = np.sqrt(k)

        weight_data = np.random.uniform(-bound, bound, size=(out_features, in_features)).astype(np.float32)
        if CUDA:
            weight_data = cp.array(weight_data)
        self.weight = Parameter(weight_data)

        if bias: 
            bias_data = np.random.uniform(-bound, bound, size=(out_features,)).astype(np.float32)
            if CUDA:
                bias_data = cp.array(bias_data)
            self.bias = Parameter(bias_data)
        else: 
            self.bias = None
        
    def forward(self, x):
        from ..functional import transpose
        weight_T = transpose(self.weight)
        output = x @ weight_T

        if self.bias is not None: 
            output = output + self.bias
        
        return output

    def __repr__(self): 
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})"