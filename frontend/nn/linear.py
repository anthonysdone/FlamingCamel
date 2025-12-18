import numpy as np
from ..tensor import Tensor, Parameter
from .module import Module

class Linear(Module): 
    def __init__(self, in_features, out_features, bias=True): 
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # kaiming he is at mit apparently woah
        k = 1.0 / in_features
        bound = np.sqrt(k)

        weight_data = np.random.uniform(-bound, bound, size=(out_features, in_features)).astype(np.float32)
        self.weight = Parameter(weight_data)

        if bias: 
            bias_data = np.random.uniform(-bound, bound, size=(out_features,)).astype(np.float32)
            self.bias = Parameter(bias_data)
        else: 
            self.bias = None
        
    def forward(self, x): 
        output = x @ self.weight.data.T

        if self.bias is not None: 
            output = output + self.bias
        
        return output

    def __repr__(self): 
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})"