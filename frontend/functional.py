import numpy as np
from .autograd import Function
from .tensor import Tensor

class Add(Function):
    @staticmethod
    def forward(ctx, a, b): 
        ctx.save_for_backward(a, b)
        return Tensor(a.data + b.data, a.requires_grad or b.requires_grad)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output

class Mul(Function): 
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return Tensor(a.data * b.data, a.requires_grad or b.requires_grad)
    
    @staticmethod
    def backward(ctx, grad_output): 
        a, b = ctx.saved_tensors
        grad_a = grad_output * b.data
        grad_b = grad_output * a.data
        return grad_a, grad_b

class MatMul(Function): 
    @staticmethod
    def forward(ctx, a, b): 
        ctx.save_for_backward(a, b)
        return Tensor(a.data @ b.data, a.requires_grad or b.requires_grad)
    
    @staticmethod
    def backward(ctx, grad_output): 
        a, b = ctx.saved_tensors
        grad_a = grad_output @ b.data.T
        grad_b = a.data.T @ grad_output
        return grad_a, grad_b

class Sum(Function):
    @staticmethod
    def forward(ctx, x): 
        ctx.save(input_shape=x.shape)
        return Tensor(np.sum(x.data), x.requires_grad)
    
    @staticmethod
    def backward(ctx, grad_output): 
        return np.ones(ctx.saved_data["input_shape"]) * grad_output
    
class ReLU(Function): 
    @staticmethod
    def forward(ctx, x): 
        ctx.save_for_backward(x)
        return Tensor(np.maximum(0, x.data), x.requires_grad)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return grad_output * (x.data > 0)
    
class CrossEntropy(Function): 
    @staticmethod
    def forward(ctx, logits, targets): 
        batch_size = logits.shape[0]

        logits_max = np.max(logits.data, axis=1, keepdims=True)
        exp_logits = np.exp(logits.data - logits_max)
        softmax = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        log_probs = np.log(softmax + 1e-8)
        correct_log_probs = log_probs[np.arange(batch_size), targets.data.astype(int)]
        loss = -np.mean(correct_log_probs)

        ctx.save_for_backward(logits, targets)
        ctx.save(softmax=softmax, batch_size=batch_size)

        return Tensor(loss, logits.requires_grad)
    
    @staticmethod
    def backward(ctx, grad_output): 
        logits, targets = ctx.saved_tensors
        softmax = ctx.saved_data["softmax"]
        batch_size = ctx.saved_data["batch_size"]

        grad_logits = softmax.copy()
        grad_logits[np.arange(batch_size), targets.data.astype(int)] -= 1
        grad_logits /= batch_size
        grad_logits *= grad_output

        return grad_logits, None

def add(a, b):
    return Add.apply(a, b)

def mul(a, b): 
    return Mul.apply(a, b)

def matmul(a, b): 
    return MatMul.apply(a, b)

def sum(x): 
    return Sum.apply(x)

def relu(x): 
    return ReLU.apply(x)

def cross_entropy(logits, targets): 
    return CrossEntropy.apply(logits, targets)