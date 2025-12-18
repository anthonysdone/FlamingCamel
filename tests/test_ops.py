import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from frontend.tensor import Tensor, tensor, zeros, ones, randn
from frontend.functional import add, mul, matmul, sum, relu, cross_entropy

def numerical_gradient(f, x, eps=1e-3): 
    grad= np.zeros_like(x.data)

    it = np.nditer(x.data, flags=["multi_index"], op_flags=["readwrite"]) # type: ignore
    while not it.finished:
        idx = it.multi_index
        old_value = x.data[idx]

        x.data[idx] = old_value + eps
        f_plus = f(x).item()

        x.data[idx] = old_value - eps
        f_minus = f(x).item()

        grad[idx] = (f_plus - f_minus) / (2 * eps)
        x.data[idx] = old_value
        it.iternext()
    
    return grad

def grad_check(f, x, eps=1e-3, atol=1e-2): 
    numerical_grad = numerical_gradient(f, x, eps)

    x.grad = None
    output = f(x)
    output.backward()
    autograd_grad = x.grad

    diff = np.abs(numerical_grad - autograd_grad)
    max_diff = np.max(diff)

    if max_diff > atol: 
        print(f"GradCheck failed! Max difference: {max_diff}")
        print(f"Numerical grad: {numerical_grad}")
        print(f"Autograd grad: {autograd_grad}")

        return False
    return True

def test_tensor_creation():
    print("\nTesting tensor creation...")

    t = zeros(2, 3)
    assert t.shape == (2, 3), f"Expected shape (2, 3), got {t.shape}"
    assert np.all(t.data == 0), f"zeros() should create all zeros"

    t = ones(2, 3)
    assert t.shape == (2, 3), f"Expected shape (2, 3), got {t.shape}"
    assert np.all(t.data == 1), f"ones() should create all ones"

    t = randn(2, 3)
    assert t.shape == (2, 3), f"Expected shape (2, 3), got {t.shape}"

    t = tensor([1.0, 2.0, 3.0], requires_grad=True)
    assert t.requires_grad == True, "requires_grad not set correctly"
    assert t.shape == (3,), f"Expected shape (3,), got {t.shape}"

    print("Tensor creation tests passed!")

def test_add_op(): 
    print("\nTesting addition...")

    a = tensor([1.0, 2.0, 3.0], requires_grad=True)
    b = tensor([4.0, 5.0, 6.0], requires_grad=True)
    c = a + b

    expected = np.array([5.0, 7.0, 9.0])
    assert np.allclose(c.data, expected), f"Expected {expected}, got {c.data}"

    def f1(x): 
        return sum(x + b)
    assert grad_check(f1, a), "Add gradcheck failed for first argument"

    def f2(x):
        return sum(a + x)
    assert grad_check(f2, b), "Add gradcheck failed for second argument"

    print("Addition test passed!")

def test_mul_op(): 
    print("\nTesting multiplication...")

    a = tensor([1.0, 2.0, 3.0], requires_grad=True)
    b = tensor([4.0, 5.0, 6.0], requires_grad=True)
    c = a * b

    expected = np.array([4.0, 10.0, 18.0])
    assert np.allclose(c.data, expected), f"Expected {expected}, got {c.data}"

    def f1(x): 
        return sum(x * b)
    assert grad_check(f1, a), "Mul gradcheck failed for first argument"

    def f2(x):
        return sum(a * x)
    assert grad_check(f2, b), "Mul gradcheck failed for second argument"

    print("Mutliplication test passed!")

def test_matmul_op():
    print("\nTesting matrix multiplication...")

    a = tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
    c = a @ b

    expected = np.array([[19.0, 22.0], [43.0, 50.0]])
    assert np.allclose(c.data, expected), f"Expected {expected}, got {c.data}"

    def f1(x): 
        return sum(x @ b)
    assert grad_check(f1, a, atol=1e-2), "Matmul gradcheck failed for first argument"

    def f2(x):
        return sum(a @ x)
    assert grad_check(f2, b, atol=1e-2), "Matmul gradcheck failed for second argument"

    print("Matrix multiplication test passed!")

def test_relu_op(): 
    print("\nTesting ReLU...")

    x = tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
    y = relu(x)

    expected = np.array([0.0, 0.0, 0.0, 1.0, 2.0])
    assert np.allclose(y.data, expected), f"Expected {expected}, got {y.data}"

    x = tensor([-2.0, -1.0, 1.0, 2.0], requires_grad=True)
    def f(x): 
        return sum(relu(x))
    assert grad_check(f, x), "ReLU gradcheck failed"

    print("ReLU test passed!")

def test_sum_op(): 
    print("\nTesting sum...")

    x = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    y = sum(x)

    expected = 21.0
    assert np.allclose(y.item(), expected), f"Expected {expected}, got {y.item()}"

    def f(x): 
        return sum(x)
    assert grad_check(f, x), "Sum gradcheck failed"

    print("Sum test passed!")

def test_cross_entropy_op():
    print("\nTesting cross entropy...")

    logits = tensor([[2.0, 1.0, 0.1], [1.0, 3.0, 0.2]], requires_grad=True)
    targets = tensor([0, 1])
    loss = cross_entropy(logits, targets)

    assert loss.shape == (), f"Expected scalar loss, got shape {loss.shape}"
    assert loss.item() > 0, "Loss should be positive"

    def f(x):
        return cross_entropy(x, targets)
    assert grad_check(f, logits, atol=1e-2), "CrossEntropy gradcheck failed"

    print("Cross entropy tests passed")

def test_compound_ops():
    print("\nTesting compound operations...")

    x = tensor([[1.0, 2.0]], requires_grad=True)
    W = tensor([[0.5, 0.3], [0.2, 0.4]], requires_grad=True)
    b = tensor([0.1, 0.2], requires_grad=True)
    g = relu(x @ W + b)

    expected = np.maximum(0, x.data @ W.data + b.data)
    assert np.allclose(g.data, expected), f"Expected {expected}, got {g.data}"

    def f_x(x):
        return sum(relu(x @ W + b))
    
    def f_W(W): 
        return sum(relu(x @ W + b))
    
    def f_b(b): 
        return sum(relu(x @ W + b))
    
    assert grad_check(f_x, x, atol=1e-2), "Compound operation gradcheck failed for x"
    assert grad_check(f_W, W, atol=1e-2), "Compound operation gradcheck failed for W"
    assert grad_check(f_b, b, atol=1e-2), "Compound operation gradcheck failed for b"

    print("Compount operations test passed!")

def test_backward_accumulation():
    print("\nTesting backward accumulation...")

    x = tensor([1.0, 2.0, 3.0], requires_grad=True)

    y1 = sum(x * 2.0)
    y1.backward()
    grad1 = x.grad.copy() # type: ignore

    y2 = sum(x * 3.0)
    y2.backward()
    grad2 = x.grad

    expected = grad1 + np.array([3.0, 3.0, 3.0])
    assert np.allclose(grad2, expected), f"Expected {expected}, got {grad2}" # type: ignore

    print("Backward accumulation test passed!")

if __name__ ==  "__main__": 
    print("\nRunning operation tests...")
    print("=" * 30)
    test_tensor_creation()
    test_add_op()
    test_mul_op()
    test_matmul_op()
    test_relu_op()
    test_sum_op()
    test_cross_entropy_op()
    test_compound_ops()
    test_backward_accumulation()
    print("\n" + "=" * 30)
    print("All tests passed!")