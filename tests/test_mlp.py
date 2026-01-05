import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from frontend.tensor import Tensor, tensor
from frontend.nn.module import Module
from frontend.nn.linear import Linear
from frontend.optim.sgd import SGD
from frontend.functional import cross_entropy

try:
    import cupy as cp
    CUDA = True
except ImportError:
    CUDA = False

class MLP(Module): 
    def __init__(self, input_size, hidden_sizes, output_size): 
        super().__init__()
        
        self.layers = []

        prev_size = input_size
        for hidden_size in hidden_sizes:
            self.layers.append(Linear(prev_size, hidden_size))
            prev_size = hidden_size

        self.layers.append(Linear(prev_size, output_size))
    
    def forward(self, x): 
        for layer in self.layers[:-1]:
            x = layer(x)
            x = x.relu()

        x = self.layers[-1](x)
        return x

def test_gradient_flow(): 
    print("\nTesting gradient flow...")
    print("=" * 30)

    model = MLP(input_size=10, hidden_sizes=[20], output_size=5)

    X = tensor(np.random.randn(4, 10).astype(np.float32), requires_grad=False)
    y = tensor(np.array([0, 1, 2, 3], dtype=np.float32))

    logits = model(X)
    loss = cross_entropy(logits, y)

    loss.backward()

    params = model.parameters()
    print(f"Total parameters: {len(params)}")

    for i, param in enumerate(params):
        assert param.grad is not None, f"Parameter {i} has no gradient"
        assert not np.all(param.grad == 0), f"Parameter {i} has zero gradient"
        print(f"Parameter {i}: shape={param.shape}, grad_mean={np.mean(np.abs(param.grad)):.6f}")
    
    print("=" * 30)
    print("Gradient flow test passed!")

def test_xor():
    print("\nTesting MLP on XOR...")
    print("=" * 30)

    X = np.array([
        [0.0, 0.0], 
        [0.0, 1.0], 
        [1.0, 0.0], 
        [1.0, 1.0],
    ], dtype=np.float32)

    y = np.array([0, 1, 1, 0], dtype=np.float32)

    model = MLP(input_size=2, hidden_sizes=[8], output_size=2)
    optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)

    epochs = 1000
    for epoch in range(epochs): 
        X_tensor = tensor(X, requires_grad=False)
        logits = model(X_tensor)

        y_tensor = tensor(y, requires_grad=False)
        loss = cross_entropy(logits, y_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")
    
    print("Final predictions:")
    X_tensor = tensor(X, requires_grad=False)
    logits = model(X_tensor)
    predictions = np.argmax(logits.data, axis=1)

    for i in range(len(X)):
        print(f"Input: {X[i]}, True: {int(y[i])}, Prediction: {predictions[i]}")
    
    accuracy = np.mean(predictions == y)
    print(f"Accuracy: {accuracy * 100:.1f}")

    assert accuracy >= 0.99, f"Expected near 100% accuracy"

    print("=" * 30)
    print("MLP XOR test passed!")

def test_mnist():
    print("\nTesting MLP on MNIST...")
    print("=" * 30)

    from torchvision import datasets, transforms

    mnist_train = datasets.MNIST(
        root="./data", 
        train=True, 
        download=True,
        transform=transforms.ToTensor()
    )

    n_samples = 100
    indices = np.random.choice(len(mnist_train), n_samples, replace=False)

    X = []
    y = []

    for idx in indices: 
        img, label = mnist_train[idx]
        X.append(img.numpy().flatten())
        y.append(label)
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    n_train = int(0.8 * len(X))
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    print(f"Train: {len(X_train)} sameples, Test: {len(X_test)} samples")

    model = MLP(input_size=784, hidden_sizes=[128, 64], output_size=10)
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)

    epochs = 100
    batch_size = 10

    for epoch in range(epochs): 
        indices = np.random.permutation(len(X_train))
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        total_loss = 0.0

        for i in range(0, len(X_train), batch_size): 
            batch_X = X_shuffled[i:i + batch_size]
            batch_y = y_shuffled[i:i + batch_size]

            X_tensor = tensor(batch_X, requires_grad=False)
            logits = model(X_tensor)

            y_tensor = tensor(batch_y, requires_grad=False)
            loss = cross_entropy(logits, y_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 20 == 0: 
            avg_loss = total_loss / (len(X_train) // batch_size)

            X_tensor = tensor(X_train, requires_grad=False)
            logits = model(X_tensor)
            predictions = np.argmax(logits.data, axis=1)
            train_acc = np.mean(predictions == y_train)

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Train Acc: {train_acc * 100:.1f}")

    X_tensor = tensor(X_test, requires_grad=False)
    logits = model(X_tensor)
    predictions = np.argmax(logits.data, axis=1)
    test_acc = np.mean(predictions == y_test)

    print(f"Final test accuracy: {test_acc * 100:.1f}%")

    assert test_acc >= 0.5, f"Expected >50% test accuracy, got {test_acc * 100:.1f}%"

    print("=" * 30)
    print("MLP MNIST test passed!")

if __name__ == "__main__": 
    print("\nRunning MLP tests...")
    print("=" * 30)
    test_gradient_flow()
    test_xor()
    test_mnist()
    print("\n" + "=" * 30)
    print("All MLP tests passed!")