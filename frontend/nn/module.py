from ..tensor import Tensor, Parameter

class Module: 
    def __call__(self, *args, **kwargs): 
        return self.forward(*args, **kwargs)
    
    def forward(self, *args, **kwargs): 
        raise NotImplementedError
    
    def parameters(self): 
        params = []
        
        for attr_name, attr in self.__dict__.items(): 
            if isinstance(attr, Parameter): 
                params.append(attr)

            elif isinstance(attr, Module): 
                params.extend(attr.parameters())
            
            elif isinstance(attr, list):
                for item in attr: 
                    if isinstance(item, Module):
                        params.extend(item.parameters())

        return params
    
    def to(self, device): 
        for param in self.parameters():
            param.data = param.to(device).data
        
        return self

    def __repr(self): 
        return f"{self.__class__.__name__}"