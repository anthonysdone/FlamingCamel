import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import cupy as cp
    CUDA = True
except ImportError:
    CUDA = False

from frontend.tensor import Tensor, tensor
from frontend.functional import add, mul, relu, sum


def grad_check(f, x, eps=1e-3, atol=1e-2): 
    is_gpu = CUDA and isinstance(x.data, cp.ndarray)
    
    if is_gpu:
        x_data_cpu = cp.asnumpy(x.data)
        x_cpu = tensor(x_data_cpu, requires_grad=True, device="cpu")
    else:
        x_cpu = x
    
    x_np = x_cpu.data.copy()
    grad = np.zeros_like(x_np)
    
    it = np.nditer(x_np, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        old_value = x_np[idx]

        x_np[idx] = old_value + eps
        x_cpu.data = x_np.copy()
        f_plus = f(x_cpu).item()

        x_np[idx] = old_value - eps
        x_cpu.data = x_np.copy()
        f_minus = f(x_cpu).item()

        grad[idx] = (f_plus - f_minus) / (2 * eps)
        x_np[idx] = old_value
        it.iternext()
    
    if is_gpu:
        x.grad = None
        output = f(x)
        output.backward()
        autograd_grad = cp.asnumpy(x.grad) if isinstance(x.grad, cp.ndarray) else x.grad
    else:
        x.grad = None
        output = f(x)
        output.backward()
        autograd_grad = x.grad

    diff = np.abs(grad - autograd_grad)
    max_diff = np.max(diff)

    if max_diff > atol: 
        print(f"GradCheck failed! Max difference: {max_diff}")
        print(f"Numerical grad: {grad}")
        print(f"Autograd grad: {autograd_grad}")

        return False
    return True


def test_add_op_gpu():
    print("\nTesting addition on CUDA...")
    
    if not CUDA:
        print("CUDA not available, skipping")
        return

    a = tensor([1.0, 2.0, 3.0], requires_grad=True, device="cuda")
    b = tensor([4.0, 5.0, 6.0], requires_grad=True, device="cuda")
    c = a + b

    expected = cp.array([5.0, 7.0, 9.0])
    assert cp.allclose(c.data, expected), f"Expected {expected}, got {c.data}"

    b_cpu = tensor([4.0, 5.0, 6.0], requires_grad=True, device="cpu")
    a_cpu = tensor([1.0, 2.0, 3.0], requires_grad=True, device="cpu")
    
    def f1(x): 
        return sum(x + b_cpu)
    assert grad_check(f1, a_cpu), "Add gradcheck failed for first argument"

    def f2(x):
        return sum(a_cpu + x)
    assert grad_check(f2, b_cpu), "Add gradcheck failed for second argument"

    print("Addition CUDA test passed!")


def test_mul_op_gpu():
    print("Testing multiplication on CUDA...")

    if not CUDA:
        print("CUDA not available, skipping")
        return

    a = tensor([1.0, 2.0, 3.0], requires_grad=True, device="cuda")
    b = tensor([4.0, 5.0, 6.0], requires_grad=True, device="cuda")
    c = a * b

    expected = cp.array([4.0, 10.0, 18.0])
    assert cp.allclose(c.data, expected), f"Expected {expected}, got {c.data}"

    b_cpu = tensor([4.0, 5.0, 6.0], requires_grad=True, device="cpu")
    a_cpu = tensor([1.0, 2.0, 3.0], requires_grad=True, device="cpu")
    
    def f1(x): 
        return sum(x * b_cpu)
    assert grad_check(f1, a_cpu), "Mul gradcheck failed for first argument"

    def f2(x):
        return sum(a_cpu * x)
    assert grad_check(f2, b_cpu), "Mul gradcheck failed for second argument"

    print("Multiplication CUDA test passed!")


def test_relu_op_gpu():
    print("Testing ReLU on CUDA...")

    if not CUDA:
        print("CUDA not available, skipping")
        return

    x = tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True, device="cuda")
    y = relu(x)

    expected = cp.array([0.0, 0.0, 0.0, 1.0, 2.0])
    assert cp.allclose(y.data, expected), f"Expected {expected}, got {y.data}"

    x_cpu = tensor([-2.0, -1.0, 1.0, 2.0], requires_grad=True, device="cpu")
    def f(x): 
        return sum(relu(x))
    assert grad_check(f, x_cpu), "ReLU gradcheck failed"

    print("ReLU CUDA test passed!")


if __name__ == "__main__":
    print("Running GPU operation tests...")
    print("=" * 30)
    
    test_add_op_gpu()
    test_mul_op_gpu()
    test_relu_op_gpu()
    
    print("\n" + "=" * 30)
    print("All GPU operation tests passed!")
