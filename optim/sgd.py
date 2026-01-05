import numpy as np

class SGD:
    def __init__(self, parameters, lr=0.01, momentum=0.0):
        self.parameters = list(parameters)
        self.lr = lr
        self.momentum = momentum

        self.velocities = []
        if self.momentum > 0: 
            for param in self.parameters: 
                velocity = np.zeros_like(param.data)
                self.velocities.append(velocity)
    
    def zero_grad(self): 
        for param in self.parameters: 
            param.grad = None
    
    def step(self): 
        for i, param in enumerate(self.parameters): 
            if param.grad is None: 
                continue
        
            grad = param.grad if isinstance(param.grad, np.ndarray) else param.grad.data

            if self.momentum > 0: 
                velocity = self.velocities[i]
                velocity = self.momentum * velocity + grad
                self.velocities[i] = velocity

                param.data -= self.lr * velocity
            else: 
                param.data -= self.lr * grad
        
        def __repr__(self): 
            return f"SGD(lr={self.lr}, momentum={self.momentum}, n_params={len(self.parameters)})"