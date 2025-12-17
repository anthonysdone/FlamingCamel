## FlamingCamel: A minimal deep-learning library for learning Metal

Below is a deliberately incremental, simplicity-first blueprint. At every stage you can train something, measure something, and optimize something—without framework complexity.

You will learn Metal GPU programming on Apple Silicon Macs. The concepts (memory hierarchies, tiling, parallelism) transfer directly to Cuda —only the API syntax differs.

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
1. Correct baseline (CPU or naive Metal)
2. Optimized Metal implementation (later)
3. Test comparing them

This keeps the library usable while you optimize kernels.

### D. No production features
No distributed training, fancy compilation, or full PyTorch semantics—this is educational.

---

## Minimal architecture

### Python surface
* `Tensor` with device dispatch
* `nn`: `Linear`, `LayerNorm`, `Embedding`, `Attention`
* `optim`: `SGD`, `Adam`
* `functional`: stateless ops (added per-stage)

### Metal shaders via Python bindings
* Memory: allocate buffers, host↔device copies via Metal API
* Compute kernels: elementwise, reductions, matmul (custom GEMM), layernorm, softmax, attention, embedding

### Core state
* `TensorImpl`: Metal buffer, shape, dtype, device flag
* Allocator: raw `MTLBuffer` allocation (no caching initially)

---

## Stage plan: always train something

Each stage ends with a working model, benchmark, and one profiler exercise (Instruments or Metal Debugger).

### Stage 0 — Pure Python CPU skeleton (~300-400 LOC)

**Purpose:** Validate autograd + API before implementing Metal kernels.

Implement (NumPy backend):
* `Tensor`, tape-based autograd
* Ops: `add`, `mul`, `matmul`, `sum`, `relu`, `cross_entropy`
* `nn.Module`, `Linear`, `MLP`
* `SGD` or `Adam`

Milestone: Train XOR or tiny MNIST via MLP. Gradcheck `matmul` and `relu`.

**Outcome:** Understand autograd dataflow.

---

### Stage 1 — GPU memory + 3 elementwise kernels (~400-600 LOC)

**Purpose:** Get "Tensor on GPU" working.

Add:
* `Tensor.to("mps")` and `.to("cpu")` (Metal Performance Shaders device)
* GPU memory management via Metal buffers
* Metal compute kernels: `add`, `mul`, `relu` (one per op, no fusion, FP32)
* Autograd uses Metal kernels when on GPU

Milestone: Same MLP trains on GPU. Profile kernel dispatch.

**Outcome:** Compute kernel dispatch, device buffers, Python ↔ GPU integration.

---

### Stage 2 — Reductions + softmax (~500-700 LOC)

**Purpose:** Unlock softmax, stable losses, learn reductions.

Add Metal kernels:
* `sum` and `max` reductions (two-pass: threadgroup-wise → final reduce)
* `softmax` forward (max-subtract stability)
* `log_softmax`, stable `cross_entropy` forward

Milestone: Classifier with softmax+loss on GPU. Profile reduction kernel.

**Outcome:** Threadgroup memory reductions, SIMD-group primitives, numerical stability.

---

### Stage 3 — Matmul: learn GEMM (~1000-1500 LOC)

**Purpose:** Learn the most important kernel in deep learning through incremental optimization.

This stage follows the optimization path from [Simon Boehm's CUDA matmul blog](https://siboehm.com/articles/22/CUDA-MMM) and [Laurent Mazare's Metal implementation](https://github.com/LaurentMazare/gemm-metal).

**Kernel 1: Naive Implementation** (~50 LOC)
* Each thread computes one output element
* Triple-loop: `C[i,j] = Σ A[i,k] * B[k,j]`
* Grid of threadgroups maps directly to output matrix
* Implement `Linear` module, test correctness
* Expected: ~100-300 GFLOPS, poor memory bandwidth utilization

**Kernel 2: Global Memory Coalescing** (~50 LOC)
* Remap threads to enable coalesced memory access
* Threads in same SIMD-group access consecutive memory locations
* Change thread indexing: consecutive `thread_index_in_threadgroup` loads consecutive columns of A
* Expected: 6-8x speedup from improved bandwidth (~600-2000 GFLOPS)

**Kernel 3: Threadgroup Memory Cache-Blocking** (~100 LOC)
* Cache tiles of A and B in threadgroup (shared) memory
* Each threadgroup loads 32×32 tiles, reuses data across threads
* Two-level tiling: outer loop over K dimension, inner dot product
* Use `threadgroup_barrier()` for synchronization
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
* Transpose A while loading into threadgroup memory
* Enables vectorized loads from threadgroup memory too
* Expected: 10-20% speedup (~15000-20000 GFLOPS)

**Kernel 7-9: Autotuning** (~200 LOC)
* Parameterize: `BM`, `BN`, `BK` (threadgroup tile sizes), `TM`, `TN` (thread tile sizes)
* Grid search over valid configurations
* Test different tile sizes: 32, 64, 128 for BM/BN; 8, 16 for BK
* Find optimal configuration for your hardware (M1/M2/M3)
* Expected: 5-10% speedup depending on matrix size

**Kernel 10: SIMD-group Tiling (Advanced, Optional)** (~200 LOC)
* Add another tiling level: SIMD-group tiles within each threadgroup
* Each SIMD-group (32 threads) computes a larger tile together
* Better register cache locality and SIMD-group level reuse
* Maps well to SIMD-group matrix operations (`simdgroup_matrix_storage`)
* Expected: 5-15% additional speedup (~20000-23000 GFLOPS)

**Kernel 11: Double Buffering (Future Work)**
* Overlap computation with memory loading
* Prefetch next tile while computing current tile
* Requires careful register management

**Implementation Notes:**
* Implement backward pass after forward works: `dW = X^T @ dY`, `dX = dY @ W^T`
* Test each kernel against previous version and NumPy
* Profile with Metal Debugger after each step
* Track: GFLOPS, memory bandwidth (GB/s), occupancy
* Compare final version against Metal Performance Shaders GEMM

**Milestones:**
* After Kernel 3: Train small MLP (slow but correct)
* After Kernel 6: Train MLP on MNIST/CIFAR at reasonable speed
* After Kernel 10: Within 90-95% of MPS GEMM performance

**Outcome:** Deep understanding of:
* Memory hierarchy (global → threadgroup → register)
* Memory coalescing and bandwidth optimization
* Arithmetic intensity and compute vs memory bound
* Multi-level tiling strategies
* Why GEMM dominates deep learning training time

---

### Stage 4 — LayerNorm + GELU (~400-600 LOC)

**Purpose:** Learn kernels essential for Transformers (simpler than attention).

Implement Metal kernels:
* `layernorm_forward` (two-pass: compute mean/var, normalize)
* `layernorm_backward` (after forward works)
* `gelu` approximation kernel

Milestone: "Transformer block" (MLP + LayerNorm, no attention). Profile memory bottlenecks.

**Outcome:** Per-element normalization, multi-pass kernels, memory bandwidth.

---

### Stage 5 — Attention (~1200-2000 LOC)

**Purpose:** Capstone Metal learning kernel.

**5A: Naive attention** (correctness baseline)

Implement:
* `scores = Q @ K^T` (MPS GEMM or custom)
* `P = softmax(scores)` (existing softmax)
* `out = P @ V` (MPS GEMM or custom)
* Backward: GEMMs + softmax backward

Milestone: GPT-mini trains end-to-end (slow, correct).

**5B: Flash attention** (~800-1200 LOC)

Implement Flash Attention 2 via incremental optimizations (Parts 1-6 from [Sonny's flash attention blog](lubits.ch/flash), adapted for Metal on Apple Silicon):

**Part 1-2: Foundation & Building Blocks**
* Understand Metal equivalents: `simdgroup_matrix_storage`, Metal async copy, threadgroup memory tiling
* Block-wise streaming: Load K/V blocks into threadgroup memory; maintain running max `m`, sum `l` per query
* Output updated incrementally (no full attention matrix materialized)
* Forward only; backward stays naive

**Part 3: Kernel 1 - Baseline Implementation** (~150-200 LOC)
* Naive Flash Attention kernel achieving ~40-50% of MPS reference
* Work distribution: threadgroup tiles (B_r=64, B_c=64), simdgroup-level parallelism
* Memory transfers: Global → Threadgroup → SIMD-group using Metal async copy primitives

**Part 4: Kernel 2 - Bank Conflicts & Swizzling** (~100-150 LOC)
* XOR-based swizzling to eliminate threadgroup memory bank conflicts
* Target: 2x performance improvement (80-100% of reference)

**Part 5: Kernel 3-5 - Metal GEMM Optimizations** (~250-350 LOC)
* Kernel 3: Double buffering global→threadgroup transfers (eager K/V loading)
* Kernel 4: Fragment interleaving for SIMD-group→register loads
* Kernel 5: Double buffering threadgroup→register transfers
* Target: ~100% of MPS reference performance

**Part 6: Kernel 6 - FP Instruction Fusion & Auto-Tuning** (~150-200 LOC)
* Fuse FP multiply-add in online softmax (reduce instruction count)
* Auto-tune block sizes: test (B_r, B_c) ∈ {(64,32), (64,64), (128,32), (128,64)}
* Target: 100-105% of MPS reference

Milestone: GPT-mini trains faster, handles 4K+ sequence lengths efficiently.

**Outcome:** 
- IO-aware algorithms, threadgroup tiling, online numerical stability
- Memory bank conflict resolution via swizzling
- Latency hiding through double buffering at multiple memory levels
- Instruction-level optimization (FMA fusion)
- Configuration space exploration (auto-tuning)

---

## Minimal operator set

Implement **only what each stage needs**:

| Stage | Required | Where |
|-------|----------|-------|
| 0 | `add`, `mul`, `matmul`, `sum`, `relu`, `cross_entropy` | NumPy |
| 1 | `add`, `mul`, `relu` | Metal compute kernels |
| 2 | `sum`, `max`, `softmax`, `log_softmax` | Metal compute kernels |
| 3 | `matmul` (naive → tiled → optimized) | Custom Metal GEMM kernels |
| 4 | `layernorm`, `gelu` | Metal compute kernels |
| 5 | `attention` | Metal kernels + MPS GEMM |

**Tensor ops:** `reshape`, `transpose` (materialize only).

**Never implement:** `sub`, `div`, `exp`, `log`, `conv2d`, `concat`—skip or decompose via autograd.

---

## What to measure at every stage

### Correctness
* CPU vs Metal on random tensors
* Gradcheck on small shapes

### Performance
* Time forward and backward for the key op (e.g., softmax, layernorm)
* Report throughput (GB/s) or wall-clock time

### Profiling
* Instruments: measure kernel dispatch overhead, memory transfers
* Metal Debugger: memory bandwidth, occupancy, threadgroup memory usage

### One optimization per stage
* Vectorized loads, layout changes, threadgroup memory reuse, fewer passes, or fusion

This keeps work focused on **Metal GPU learning, not framework plumbing**.

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
  backend.py    # Metal device management and kernel dispatch
  nn/
    module.py         # Module base class
    linear.py
    layernorm.py
    embedding.py
    attention.py
  optim/
    sgd.py
    adam.py
backend/shaders
  ops_elementwise.metal  # add, mul, relu
  ops_reduce.metal       # sum, max
  ops_softmax.metal      # softmax, log_softmax
  ops_matmul.metal       # GEMM (naive, tiled, optimized)
  ops_layernorm.metal    # layernorm
  ops_attention.metal    # attention
tests/
  test_ops.py         # CPU vs Metal correctness
  test_gradcheck.py   # Gradient numerical checks
examples/
  mlp.py              # MLP training
  gpt_mini.py         # GPT-mini with attention
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
model = MLP(784, 256, 10).to("mps")
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
    for x_batch, y_batch in dataloader:
        x = camel.tensor(x_batch).to("mps")
        y = camel.tensor(y_batch).to("mps")
        
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
).to("mps")

optimizer = camel.optim.Adam(model.parameters(), lr=3e-4)

# Training
for tokens in dataloader:
    tokens = camel.tensor(tokens).to("mps")  # (B, T)
    
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
* **Device-agnostic:** Same code works on CPU or MPS (Metal Performance Shaders)
* **Explicit control:** No magic—you see every forward and backward pass

---

## Natural stopping points

Each stage delivers a coherent, trained model:

* **After Stage 1:** GPU MLP works.
* **After Stage 2:** GPU MLP + stable softmax loss.
* **After Stage 3:** Fast, practical MLP (MPS GEMM).
* **After Stage 4:** Transformer blocks (no attention).
* **After Stage 5A:** GPT-mini trains end-to-end.
* **After Stage 5B:** GPT-mini optimized (flash attention).

---

## Next steps

1. **Specify your target chip** (e.g., M1, M2, M3, M4) so we can tailor kernel priorities
2. **Define GPT-mini scope** (parameter count, sequence length)
3. **Start Stage 0** (CPU skeleton) immediately on macOS

From there, each stage adds ~300-2000 LOC depending on complexity.

// ...existing code...

## Next steps

1. **Specify your target chip** (e.g., M1, M2, M3, M4) so we can tailor kernel priorities
2. **Define GPT-mini scope** (parameter count, sequence length)
3. **Start Stage 0** (CPU skeleton) immediately on macOS

From there, each stage adds ~300-2000 LOC of focused Metal learning that transfers directly to CUDA.