## FlamingCamel: A minimal deep-learning library for learning CUDA

Below is a deliberately incremental, simplicity-first blueprint. At every stage you can train something, measure something, and optimize something—without framework complexity.

You will learn CUDA GPU programming on NVIDIA GPUs using Modal for cloud compute. The concepts (memory hierarchies, tiling, parallelism) are fundamental to GPU programming.

---

## Guiding constraints

Non-negotiable for steady progress:

### A. Tensor model
* Only **dense**, **contiguous** tensors initially.
* No complex broadcasting—implement the minimum.
* Limited ops: only what you need for MLP/Transformer/GPT.
* Start with FP32 only.

### B. Autograd
* Reverse-mode with a tape.
* No higher-order gradients, in-place ops, or dynamic shapes.

### C. Reference path
For every op:
1. Correct baseline (CPU or naive CUDA)
2. Optimized CUDA implementation (later)
3. Test comparing them

This keeps the library usable while you optimize kernels.

### D. No production features
No distributed training, fancy compilation, or full PyTorch semantics—this is educational.

---

## Minimal architecture

### Python surface (frontend/)
* `Tensor` with device dispatch
* `nn`: `Linear`, `LayerNorm`, `Embedding`, `Attention`
* `optim`: `SGD`, `Adam`
* `functional`: stateless ops (added per-stage)

### CUDA kernels (backend/) via Python bindings
* **Python-CUDA Bridge:** Use `cupy.RawKernel` for loading and calling CUDA kernels
  - Compile `.cu` files with `nvcc` to `.cubin` or `.ptx`
  - Load compiled kernels: `kernel = cupy.RawKernel(code, 'kernel_name')`
  - Launch kernels: `kernel((grid,), (block,), (args,))`
* **Memory Management:** Use `cupy.ndarray` to wrap CUDA device pointers
  - Automatic memory management via Python garbage collection
  - Interop with NumPy via `.get()` (device→host) and `cupy.asarray()` (host→device)
* **Compute kernels:** elementwise, reductions, matmul (custom GEMM), layernorm, softmax, attention

### Core state
* `TensorImpl`: 
  - Wraps `cupy.ndarray` for CUDA or `numpy.ndarray` for CPU
  - Store: `.data` (array), `.shape`, `.dtype`, `.device`, `.requires_grad`
  - Device pointer accessible via `cupy.ndarray.data.ptr`
* **Memory Management:**
  - Leverage cupy's automatic memory management (no manual cudaMalloc/cudaFree)
  - `.to('cuda')` creates cupy array via `cupy.asarray(numpy_data)`
  - `.to('cpu')` extracts to NumPy via `cupy_array.get()`
* **Autograd Integration:**
  - Each `Function.forward()` stores necessary tensors in `ctx` (CPU or GPU)
  - `Function.backward()` dispatches to CUDA kernels if input.device == 'cuda'
  - Gradients accumulate on same device as tensor (.grad has same device as tensor)

---

## Stage plan: always train something

Each stage ends with a working model, benchmark, and one profiler exercise (Nsight Systems or Nsight Compute).

### Stage 0 — Pure Python CPU skeleton (~300-400 LOC)

**Purpose:** Validate autograd + API before implementing CUDA kernels.

Implement (NumPy backend):
* `Tensor`, tape-based autograd
* Ops: `add`, `mul`, `matmul`, `sum`, `relu`, `cross_entropy`
* `nn.Module`, `Linear`, `MLP`
* At least one optimizer: `SGD` (simpler) or `Adam` (both optional)

Milestone: Train XOR or tiny MNIST via MLP. Gradcheck `matmul` and `relu`.

**Outcome:** Understand autograd dataflow.

---

### Stage 1 — GPU memory + 3 elementwise kernels (~400-600 LOC)

**Purpose:** Get "Tensor on GPU" working.

**Development Setup Options:**

**Option A: Local CUDA (Recommended for this project)**
* Install CUDA Toolkit and nvcc locally
* Use `cupy` for Python-CUDA bridge (simplifies kernel calls)
* Direct kernel compilation and execution
* Low latency for training loops

**Option B: Modal for Learning/Prototyping (Limited)**
* Install Modal and authenticate: `pip install modal`
* Create `modal_run.py` with CUDA image
* Specify GPU type (A100, T4, etc.)
* Use for: kernel prototyping, benchmarking, testing
* **Limitation:** High latency for training loops (data transfer overhead)
* **Best for:** Learning CUDA concepts, not production training

**This guide assumes Option A (local CUDA) for practical development.**

Add:
* `Tensor.to("cuda")` and `.to("cpu")` for device management
* GPU memory management via CUDA (cudaMalloc, cudaMemcpy)
* CUDA kernels: `add`, `mul`, `relu` (one per op, no fusion, FP32)
* Autograd uses CUDA kernels when on GPU

Milestone: Same MLP trains on GPU. Profile kernel dispatch.

**Outcome:** Kernel launch, device memory, Python ↔ GPU integration.

---

### Stage 2 — Reductions + softmax (~500-700 LOC)

**Purpose:** Unlock softmax, stable losses, learn reductions.

Add CUDA kernels:
* `sum` and `max` reductions (two-pass: block-wise → final reduce)
* `softmax` forward (max-subtract stability)
* `log_softmax`, stable `cross_entropy` forward

Milestone: Classifier with softmax+loss on GPU. Profile reduction kernel.

**Outcome:** Shared memory reductions, warp primitives, numerical stability.

---

### Stage 3 — Matmul: learn GEMM (~1000-1500 LOC)

**Purpose:** Learn the most important kernel in deep learning through incremental optimization.

This stage follows exactly [Simon Boehm's CUDA matmul blog](https://siboehm.com/articles/22/CUDA-MMM).

**Kernel 1: Naive Implementation** (~50 LOC)
* Each thread computes one output element
* Triple-loop: `C[i,j] = Σ A[i,k] * B[k,j]`
* Grid of blocks maps directly to output matrix
* Implement `Linear` module, test correctness
* Expected: ~100-300 GFLOPS, poor memory bandwidth utilization

**Kernel 2: Global Memory Coalescing** (~50 LOC)
* Remap threads to enable coalesced memory access
* Threads in same warp access consecutive memory locations
* Change thread indexing: consecutive `threadIdx.x` loads consecutive columns of A
* Expected: 6-8x speedup from improved bandwidth (~600-2000 GFLOPS)

**Kernel 3: Shared Memory Cache-Blocking** (~100 LOC)
* Cache tiles of A and B in shared memory
* Each block loads 32×32 tiles, reuses data across threads
* Two-level tiling: outer loop over K dimension, inner dot product
* Use `__syncthreads()` for synchronization
* Expected: Small improvement (~2000-3000 GFLOPS), sets up next optimization

**Kernel 4: 1D Blocktiling** (~100 LOC)
* Each thread computes multiple (e.g., 8) output elements in a column
* Increases arithmetic intensity (FLOPs per byte)
* Cache values from B in registers, reuse across multiple outputs
* Expected: 2-3x speedup (~6000-9000 GFLOPS)

**Kernel 5: 2D Blocktiling** (~150 LOC)
* Each thread computes an 8×8 tile of outputs
* Outer product in registers: load column of A, row of B
* Maximizes register reuse and arithmetic intensity
* Expected: 2x speedup (~12000-18000 GFLOPS), approaching compute bound

**Kernel 6: Vectorized Memory Access** (~100 LOC)
* Use `float4` for 128-bit vectorized loads from global memory
* Transpose A while loading into shared memory
* Enables vectorized loads from shared memory too
* Expected: 10-20% speedup (~15000-20000 GFLOPS)

**Kernel 7-9: Autotuning** (~200 LOC)
* Parameterize: `BM`, `BN`, `BK` (block tile sizes), `TM`, `TN` (thread tile sizes)
* Grid search over valid configurations
* Test different tile sizes: 32, 64, 128 for BM/BN; 8, 16 for BK
* Find optimal configuration for your hardware (A100, 4090, etc.)
* Expected: 5-10% speedup depending on matrix size

**Kernel 10: Warptiling (Advanced, Optional)** (~200 LOC)
* Add another tiling level: warp tiles within each block
* Each warp (32 threads) computes a larger tile together
* Better register cache locality and warp-level reuse
* Maps well to tensor cores (wmma API) for future work
* Expected: 5-15% additional speedup (~20000-23000 GFLOPS)

**Kernel 11: Double Buffering (Future Work)**
* Overlap computation with memory loading
* Prefetch next tile while computing current tile
* Requires careful register management

**Implementation Notes:**
* Implement backward pass after forward works: `dW = X^T @ dY`, `dX = dY @ W^T`
* Test each kernel against previous version and NumPy
* Profile with Nsight Compute after each step
* Track: GFLOPS, memory bandwidth (GB/s), occupancy
* Compare final version against cuBLAS

**Milestones:**
* After Kernel 3: Train small MLP (slow but correct)
* After Kernel 6: Train MLP on MNIST/CIFAR at reasonable speed
* After Kernel 10: Within 90-95% of cuBLAS performance

**Outcome:** Deep understanding of:
* Memory hierarchy (global → shared → register)
* Memory coalescing and bandwidth optimization
* Arithmetic intensity and compute vs memory bound
* Multi-level tiling strategies
* Why GEMM dominates deep learning training time

---

### Stage 4 — LayerNorm + GELU (~400-600 LOC)

**Purpose:** Learn kernels essential for Transformers (simpler than attention).

Implement CUDA kernels:
* `layernorm_forward` (two-pass: compute mean/var, normalize)
* `layernorm_backward` (after forward works)
* `gelu` approximation kernel

Milestone: "Transformer block" (MLP + LayerNorm, no attention). Profile memory bottlenecks.

**Outcome:** Per-element normalization, multi-pass kernels, memory bandwidth.

---

### Stage 5 — Attention (~1200-2000 LOC)

**Purpose:** Capstone CUDA learning kernel.

**5A: Naive attention** (correctness baseline)

Implement:
* `scores = Q @ K^T` (cuBLAS or custom)
* `P = softmax(scores)` (existing softmax)
* `out = P @ V` (cuBLAS or custom)
* Backward: GEMMs + softmax backward

Milestone: GPT-mini trains end-to-end (slow, correct).

**5B: Flash attention** (~800-1200 LOC)

Implement Flash Attention 2 via incremental optimizations following [lubits.ch/flash](https://lubits.ch/flash). The source is a 10-part series; we'll implement through Part 6, covering 6 kernel iterations that achieve near-optimal performance:

**Part 1-2: Foundation & Building Blocks** (Preparation, no kernel implementation)
* Understand CUDA APIs: `wmma` fragments, `cp.async` for async copies, shared memory tiling
* Block-wise streaming: Load K/V blocks into shared memory; maintain running max `m`, sum `l` per query
* Output updated incrementally (no full attention matrix materialized)
* Forward only; backward stays naive

**Part 3: Kernel 1 - Baseline Implementation** (~150-200 LOC)
* Naive Flash Attention kernel achieving ~40-50% of cuBLAS reference
* Work distribution: block tiles (B_r=64, B_c=64), warp-level parallelism
* Memory transfers: Global → Shared → Warp registers using `cp.async`

**Part 4: Kernel 2 - Bank Conflicts & Swizzling** (~100-150 LOC)
* XOR-based swizzling to eliminate shared memory bank conflicts
* Target: 2x performance improvement (80-100% of reference)

**Part 5: Kernels 3-5 - CUDA GEMM Optimizations** (~250-350 LOC)
* **Kernel 3:** Double buffering global→shared transfers (eager K/V loading with `cp.async`)
* **Kernel 4:** Fragment interleaving for warp→register loads (`ldmatrix`)
* **Kernel 5:** Double buffering shared→register transfers
* Target: ~100% of cuBLAS reference performance
* Note: Part 5 covers three separate kernel implementations (Kernels 3, 4, and 5)

**Part 6: Kernel 6 - FP Instruction Fusion & Auto-Tuning** (~150-200 LOC)
* Fuse FP multiply-add in online softmax (reduce instruction count)
* Auto-tune block sizes: test (B_r, B_c) ∈ {(64,32), (64,64), (128,32), (128,64)}
* Target: 100-105% of cuBLAS reference

Milestone: GPT-mini trains faster, handles 4K+ sequence lengths efficiently.

**Outcome:** 
- IO-aware algorithms, shared memory tiling, online numerical stability
- Memory bank conflict resolution via swizzling
- Latency hiding through double buffering at multiple memory levels (`cp.async`)
- Instruction-level optimization (FMA fusion)
- Configuration space exploration (auto-tuning)

---

### Stage 6 — CNN Layers: Convolution + Pooling (~1500-2000 LOC)

**Purpose:** Learn spatial convolution operations, essential for image processing and CNNs.

This stage follows the optimization guide from [UIC CS525 CUDA Convolution](https://www.evl.uic.edu/sjames/cs525/final.html).

**6A: Naive CPU Implementation + Basic CUDA**

Implement supporting operations:
* Add `reshape`, `transpose` to Tensor (if not already present from Stage 5)
* Add `pad` operation for boundary handling

Implement Conv2d layer:
* `nn.Conv2d` with learnable kernels
* Support configurable: kernel size, stride, padding, channels
* Forward: im2col + matmul approach for correctness baseline
* Backward: gradient w.r.t. input, weight, bias

Implement pooling layers:
* `nn.MaxPool2d` with configurable kernel size and stride
* Forward: sliding window maximum
* Backward: gradient routing to max positions

**Step 0: Naive CUDA Convolution** (~150-200 LOC)
* Direct 2D convolution kernel
* Each thread computes one output pixel
* Nested loops over kernel dimensions
* Global memory access for every kernel element
* Handle boundary conditions with zero-padding
* Expected: Very slow (~1750ms for 1024x1024), but correct

Milestone: Small CNN (Conv→ReLU→Pool→Conv→Linear) trains on MNIST (slow).

**6B: Optimized CUDA Kernels** (~1300-1800 LOC)

**Step 1: Shared Memory Caching** (~200-250 LOC)
* Cache input tiles in shared memory (e.g., 32×32)
* Each thread loads 4 values (including apron regions)
* Block size 16×16, shared memory 32×32 for kernel radius 8
* Reduce redundant global memory accesses
* Use `__syncthreads()` after loading shared memory
* Expected: 2.8× speedup (~2400ms → ~670ms for 2048×2048)

**Step 2: Separable Convolution** (~300-350 LOC)
* Decompose 2D convolution: row-wise then column-wise
* Reduces complexity from O(k²) to O(2k) per pixel
* Implement separate row and column kernels
* Row kernel: only horizontal apron needed
* Column kernel: only vertical apron needed
* Store intermediate result between passes
* Expected: 6.2× speedup over Step 1 (~370ms for 2048×2048)

**Step 3: Memory Access Optimization** (~250-300 LOC)
* Reorganize shared memory from 2D to 1D layout
* Ensure coalesced memory access patterns
* Eliminate shared memory bank conflicts
* Change indexing to consecutive access by warp threads
* Use `__mul24` for faster integer multiplication
* Expected: 3.2× speedup over Step 2 (~118ms for 2048×2048)
* **Total improvement: 57× faster than naive (2048×2048)**

**Step 4: Advanced Optimizations (Optional)** (~200-300 LOC)
* Texture memory for cached input reads
* Further block size tuning and register optimization
* Loop unrolling with `#pragma unroll`
* Expected: Additional 2× speedup (~58ms for 2048×2048)

**Pooling Kernels** (~200-250 LOC)

Implement MaxPool2d CUDA kernels:
* Forward: each thread computes one output position
* Track max value and its position (for backward)
* Use shared memory for input tile caching
* Backward: scatter gradients to max positions only
* Handle stride and padding correctly

**Testing & Validation**
* Test each optimization step against previous version
* Compare against PyTorch Conv2d for correctness
* Test various configurations: kernel sizes (3×3, 5×5, 7×7), channels, stride, padding
* Gradcheck convolution and pooling operations
* Profile with Nsight Compute after each step

**Integration & Examples**

Create example CNN architectures:
* `examples/lenet.py`: Classic LeNet-5 for MNIST
  - Conv(1→6) → ReLU → Pool → Conv(6→16) → ReLU → Pool → FC(400→120) → FC(120→84) → FC(84→10)
  - Train on MNIST, verify convergence
* `examples/simple_cnn.py`: Small CNN for CIFAR-10
  - Conv→BatchNorm→ReLU→Pool pattern
  - Compare training time: naive vs optimized kernels

**Profiling & Benchmarking**
* Profile each kernel variant with Nsight Compute
* Measure: kernel execution time, memory bandwidth, occupancy, bank conflicts
* Compare against cuDNN convolution (target: 80-90% of cuDNN performance)
* Benchmark on different image sizes: 32×32, 64×64, 128×128, 256×256
* Document performance scaling with different kernel sizes and channel counts

**Milestones:**
* After Step 0-1: LeNet trains on MNIST (slow but correct)
* After Step 2-3: Practical CNN training speed
* After Step 4: Near-optimal performance, within 80-90% of cuDNN

**Outcome:** Deep understanding of:
* Spatial convolution algorithms and optimizations
* Separable convolution decomposition
* Shared memory tiling for 2D data with halos (apron regions)
* Memory layout optimization for coalesced access
* Bank conflict resolution in 2D shared memory
* Practical CNN architectures and training
* Difference between dense (matmul) and sparse (conv) operations

---

## Minimal operator set

Implement **only what each stage needs**:

| Stage | Required | Where |
|-------|----------|-------|
| 0 | `add`, `mul`, `matmul`, `sum`, `relu`, `cross_entropy` | NumPy |
| 1 | `add`, `mul`, `relu` | CUDA kernels |
| 2 | `sum`, `max`, `softmax`, `log_softmax` | CUDA kernels |
| 3 | `matmul` (naive → tiled → optimized) | Custom CUDA GEMM kernels |
| 4 | `layernorm`, `gelu` | CUDA kernels |
| 5 | `attention` | CUDA kernels + cuBLAS |
| 6 | `conv2d`, `maxpool2d` | CUDA kernels (naive → separable → optimized) |

**Tensor ops:** `reshape`, `transpose`, `pad` (materialize only).

**Never implement:** `sub`, `div`, `exp`, `log`, `conv2d`, `concat`—skip or decompose via autograd.

---

## What to measure at every stage

### Correctness
* CPU vs CUDA on random tensors
* Gradcheck on small shapes

### Performance
* Time forward and backward for the key op (e.g., softmax, layernorm)
* Report throughput (GB/s) or wall-clock time

### Profiling
* Nsight Systems: measure kernel dispatch overhead, memory transfers
* Nsight Compute: memory bandwidth, occupancy, shared memory usage

### One optimization per stage
* Vectorized loads, layout changes, shared memory reuse, fewer passes, or fusion

This keeps work focused on **CUDA GPU learning, not framework plumbing**.

---

## Keeping the project minimal

### Module system
* `Module.parameters()` walks `__dict__` for `Parameter` instances
* `Parameter = Tensor(requires_grad=True)`

### Autograd
* Each op creates a node: (inputs, output, backward_fn, saved_data)
* No higher-order differentiation

### Tensor semantics
* No complex broadcasting initially
* No fancy slicing (only what embedding needs: 1D gather)
* No view aliasing (materialize copies)

---

## Repository layout

```
frontend/
  tensor.py           # Tensor class + autograd glue
  autograd.py         # Function, Context, backward tape
  functional.py       # Stateless ops
  backend.py          # CUDA device management and kernel dispatch
  nn/
    module.py         # Module base class
    linear.py
    layernorm.py
    embedding.py
    attention.py
    conv2d.py         # 2D convolution layer
    pooling.py        # MaxPool2d, AvgPool2d
  optim/
    sgd.py
    adam.py
backend/
  ops_elementwise.cu  # add, mul, relu
  ops_reduce.cu       # sum, max
  ops_softmax.cu      # softmax, log_softmax
  ops_matmul.cu       # GEMM (naive, tiled, optimized)
  ops_layernorm.cu    # layernorm
  ops_attention.cu    # attention
  ops_conv2d.cu       # 2D convolution (naive, shared mem, separable, optimized)
  ops_pooling.cu      # max pooling, avg pooling
tests/
  mlp.py              # MLP training
  gpt_mini.py         # GPT-mini with attention
examples/
  lenet.py            # LeNet-5 for MNIST
  simple_cnn.py       # Small CNN for CIFAR-10
modal_run.py          # Modal script
```

This is enough structure without overengineering.

---

## Usage examples

### Example 1: Simple MLP

```python
import camel
from camel import nn, optim

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x).relu()
        x = self.fc2(x).relu()
        x = self.fc3(x)
        return x

# Training loop
model = MLP(784, 256, 10).to("cuda")
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
    for x_batch, y_batch in dataloader:
        x = camel.tensor(x_batch).to("cuda")
        y = camel.tensor(y_batch).to("cuda")
        
        # Forward
        logits = model(x)
        loss = camel.functional.cross_entropy(logits, y)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Loss: {loss.item():.4f}")
```

### Example 2: GPT-mini

```python
import camel
from camel import nn

class GPTBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiHeadAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )
    
    def forward(self, x):
        # Attention block with residual
        x = x + self.attn(self.ln1(x))
        # MLP block with residual
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, max_seq_len):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([GPTBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
    
    def forward(self, idx):
        B, T = idx.shape
        
        # Embeddings
        tok_emb = self.token_emb(idx)  # (B, T, d_model)
        pos = camel.arange(T).to(idx.device)
        pos_emb = self.pos_emb(pos)     # (T, d_model)
        x = tok_emb + pos_emb            # Broadcast add
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Output head
        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab_size)
        return logits

# Usage
model = GPT(
    vocab_size=50257,
    d_model=768,
    n_layers=12,
    n_heads=12,
    max_seq_len=1024
).to("cuda")

optimizer = camel.optim.Adam(model.parameters(), lr=3e-4)

# Training
for tokens in dataloader:
    tokens = camel.tensor(tokens).to("cuda")  # (B, T)
    
    # Predict next token
    logits = model(tokens[:, :-1])  # (B, T-1, vocab_size)
    targets = tokens[:, 1:]          # (B, T-1)
    
    loss = camel.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), 
                                          targets.reshape(-1))
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Key API design principles

* **PyTorch-like syntax:** Familiar `.to()`, `.backward()`, method chaining
* **Minimal boilerplate:** Module system handles parameter registration automatically
* **Device-agnostic:** Same code works on CPU or CUDA
* **Explicit control:** No magic—you see every forward and backward pass

---

## Natural stopping points

Each stage delivers a coherent, trained model:

* **After Stage 1:** GPU MLP works.
* **After Stage 2:** GPU MLP + stable softmax loss.
* **After Stage 3:** Fast, practical MLP (cuBLAS-level).
* **After Stage 4:** Transformer blocks (no attention).
* **After Stage 5A:** GPT-mini trains end-to-end.
* **After Stage 5B:** GPT-mini optimized (flash attention).

---

## Next steps

1. **Specify your target GPU** (e.g., A100, 4090, H100) on Modal so we can tailor kernel priorities
2. **Define GPT-mini scope** (parameter count, sequence length)
3. **Start Stage 0** (CPU skeleton) locally, then move to Modal for CUDA stages

From there, each stage adds ~300-2000 LOC depending on complexity.

// ...existing code...

## Next steps

1. **Specify your target GPU** (e.g., A100, 4090, H100) on Modal so we can tailor kernel priorities
2. **Define GPT-mini scope** (parameter count, sequence length)
3. **Start Stage 0** (CPU skeleton) locally, then move to Modal for CUDA stages

From there, each stage adds ~300-2000 LOC of focused CUDA learning.