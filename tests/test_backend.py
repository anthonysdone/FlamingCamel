from frontend.backend import *
import cupy as cp

def test_cuda_add():
    print("\Testing cuda add...")
    print("=" * 30)

    a = cp.array([1, 2, 3], dtype=cp.float32)
    b = cp.array([4, 5, 6], dtype=cp.float32)
    out = cp.zeros(3, dtype=cp.float32)

    cuda_add(a, b, out)
    assert cp.all(out == cp.array([5, 7, 9], dtype=cp.float32)), f"cuda_add gave {out} instead of [5, 7, 9]"
    print("=" * 30)
    print("Gradient flow test passed!")

if __name__ ==  "__main__": 
    print("\nRunning backend tests...")
    print("=" * 30)
    test_cuda_add()
    print("\n" + "=" * 30)
    print("All tests passed!")