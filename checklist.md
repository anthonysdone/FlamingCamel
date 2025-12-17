
## Complete Project Checklist

### Stage 0: Pure Python CPU Skeleton (~300-400 LOC)

#### Core Autograd System
- [x] Create `core/autograd.py`
  - [x] Implement `Function` base class with `forward()` and `backward()` abstract methods
  - [x] Implement `Context` class for saving tensors/data needed in backward pass
  - [x] Implement backward tape/graph structure
  - [x] Add gradient accumulation logic

- [x] Create `core/tensor.py`
  - [x] Implement `Tensor` class wrapping NumPy arrays
  - [x] Add `.requires_grad` flag
  - [x] Add `.grad` attribute for storing gradients
  - [x] Implement `.backward()` method that walks the computation tape
  - [x] Add `.item()` for scalar extraction
  - [x] Add `.numpy()` for NumPy conversion
  - [x] Add `.shape`, `.dtype` properties
  - [x] Implement basic tensor creation: `zeros`, `ones`, `randn`, `arange`

#### Functional Operations
- [x] Create `core/functional.py`
  - [x] Implement `Add` function (forward/backward)
  - [x] Implement `Mul` function (forward/backward)
  - [x] Implement `MatMul` function (forward/backward)
  - [x] Implement `Sum` function (forward/backward)
  - [x] Implement `ReLU` function (forward/backward)
  - [x] Implement `CrossEntropy` function (forward/backward)
  - [x] Add convenience wrappers: `add()`, `mul()`, `matmul()`, etc.

#### Neural Network Primitives
- [ ] Create `core/nn/module.py`
  - [ ] Implement `Module` base class
  - [ ] Add `__call__()` that invokes `forward()`
  - [ ] Implement `.parameters()` method that recursively finds all Parameters
  - [ ] Implement `.to(device)` method
  - [ ] Implement `Parameter` class (Tensor with `requires_grad=True`)

- [ ] Create `core/nn/linear.py`
  - [ ] Implement `Linear` layer with weight initialization
  - [ ] Add optional bias parameter
  - [ ] Implement forward pass: `y = x @ W^T + b`

#### Optimizers
- [ ] Create `core/optim/sgd.py`
  - [ ] Implement `SGD` optimizer
  - [ ] Add `.zero_grad()` method
  - [ ] Add `.step()` method with learning rate
  - [ ] Add momentum support

- [ ] Create `core/optim/adam.py`
  - [ ] Implement `Adam` optimizer
  - [ ] Add first and second moment estimates
  - [ ] Implement bias correction
  - [ ] Add `.zero_grad()` and `.step()` methods

#### Testing & Validation
- [ ] Create `tests/test_ops.py`
  - [ ] Test basic tensor operations
  - [ ] Test autograd correctness
  - [ ] Implement numerical gradient checking

- [ ] Create `examples/mlp.py`
  - [ ] Implement simple MLP class
  - [ ] Train on XOR or tiny MNIST
  - [ ] Verify gradients are correct

**Stage 0 Milestone:** ✅ MLP trains on CPU with NumPy backend

---

### Stage 1: GPU Memory + Elementwise Kernels (~400-600 LOC)

#### Build System
- [ ] Set up Metal shader compilation pipeline
- [ ] Configure `.metal` file compilation to `.metallib`
- [ ] Create Python bindings using `pyobjc` or ctypes for Metal API

#### Metal Backend Infrastructure
- [ ] Create `core/backend.py`
  - [ ] Initialize Metal device (`MTLCreateSystemDefaultDevice`)
  - [ ] Create command queue
  - [ ] Load compiled shader library (`.metallib`)
  - [ ] Implement buffer allocation (`MTLBuffer`)
  - [ ] Implement buffer deallocation
  - [ ] Implement host-to-device copy (`buffer.contents()`)
  - [ ] Implement device-to-host copy
  - [ ] Add error checking utilities

#### Tensor Device Support
- [ ] Update `core/tensor.py`
  - [ ] Add `.device` attribute ("cpu" or "mps")
  - [ ] Implement `.to("mps")` method
  - [ ] Implement `.to("cpu")` method
  - [ ] Add device dispatch in operations

#### Elementwise Kernels
- [ ] Create `shaders/ops_elementwise.metal`
  - [ ] Implement Metal `add` kernel (element-wise)
  - [ ] Implement Metal `mul` kernel (element-wise)
  - [ ] Implement Metal `relu` kernel (element-wise)
  - [ ] Add forward pass compute functions
  - [ ] Add backward pass compute functions (gradients)

- [ ] Update `core/functional.py`
  - [ ] Add device dispatch to `add`, `mul`, `relu`
  - [ ] Call Metal kernels when tensors are on MPS
  - [ ] Keep NumPy path for CPU tensors

#### Testing
- [ ] Create `tests/test_gpu_ops.py`
  - [ ] Test CPU vs MPS correctness for `add`, `mul`, `relu`
  - [ ] Test host-device data transfers
  - [ ] Test gradient correctness on MPS

- [ ] Update `examples/mlp.py`
  - [ ] Add `.to("mps")` to model and data
  - [ ] Verify training works on GPU

#### Profiling
- [ ] Profile kernel dispatch overhead with Instruments
- [ ] Measure memory transfer times
- [ ] Document performance baseline

**Stage 1 Milestone:** ✅ MLP trains on GPU with custom Metal kernels

---

### Stage 2: Reductions + Softmax (~500-700 LOC)

#### Reduction Kernels
- [ ] Create `shaders/ops_reduce.metal`
  - [ ] Implement threadgroup-wise `sum` reduction kernel
  - [ ] Implement final `sum` reduction kernel (two-pass)
  - [ ] Implement threadgroup-wise `max` reduction kernel
  - [ ] Implement final `max` reduction kernel (two-pass)
  - [ ] Add threadgroup memory synchronization
  - [ ] Implement SIMD-group primitives

#### Softmax Kernels
- [ ] Create `shaders/ops_softmax.metal`
  - [ ] Implement `softmax` forward kernel (max-subtract for stability)
  - [ ] Implement `log_softmax` forward kernel
  - [ ] Implement `softmax` backward kernel
  - [ ] Implement `log_softmax` backward kernel

#### Functional API
- [ ] Update `core/functional.py`
  - [ ] Add `sum()` operation with Metal dispatch
  - [ ] Add `max()` operation with Metal dispatch
  - [ ] Add `softmax()` operation
  - [ ] Add `log_softmax()` operation
  - [ ] Add stable `cross_entropy()` using log_softmax

#### Testing
- [ ] Test reduction correctness (CPU vs MPS)
- [ ] Test softmax numerical stability
- [ ] Test cross_entropy gradients
- [ ] Gradcheck all new operations

#### Example
- [ ] Create `examples/classifier.py`
  - [ ] Train classifier with softmax + cross-entropy loss
  - [ ] Verify stable training

#### Profiling
- [ ] Profile reduction kernel with Metal Debugger
- [ ] Measure threadgroup memory usage
- [ ] Analyze thread divergence
- [ ] Optimize threadgroup size

**Stage 2 Milestone:** ✅ Classifier with stable softmax+loss on GPU

---

### Stage 3: Matrix Multiplication (~1000-1500 LOC)

Following https://siboehm.com/articles/22/CUDA-MMM and https://github.com/LaurentMazare/gemm-metal

#### Kernel 1: Naive Implementation
- [ ] Create `shaders/ops_matmul.metal`
  - [ ] Implement naive GEMM: each thread computes one output element
  - [ ] Triple-loop over K dimension
  - [ ] Test correctness vs NumPy on small matrices
  - [ ] Implement backward pass: `dW = X^T @ dY`, `dX = dY @ W^T`

- [ ] Update `core/nn/linear.py`
  - [ ] Use Metal matmul when on MPS
  - [ ] Verify backward pass correctness

- [ ] Benchmark and profile
  - [ ] Measure GFLOPS (expect ~100-300)
  - [ ] Check memory bandwidth utilization (should be very low)

#### Kernel 2: Global Memory Coalescing
- [ ] Remap thread indexing for coalesced access
  - [ ] Change from 2D `thread_position_in_threadgroup` to 1D
  - [ ] Ensure consecutive threads access consecutive memory
  - [ ] Threads in same SIMD-group load contiguous A elements

- [ ] Benchmark and profile
  - [ ] Measure GFLOPS (expect 6-8x speedup)
  - [ ] Verify improved memory bandwidth (should increase significantly)

#### Kernel 3: Threadgroup Memory Cache-Blocking
- [ ] Implement threadgroup memory caching
  - [ ] Allocate threadgroup memory for A and B tiles (e.g., 32×32 each)
  - [ ] Outer loop: advance through K dimension
  - [ ] Load tiles cooperatively from global to threadgroup memory
  - [ ] Use `threadgroup_barrier()` before and after computation
  - [ ] Inner loop: compute partial dot products using threadgroup data

- [ ] Benchmark and profile
  - [ ] Measure GFLOPS (expect small improvement, ~2000-3000)
  - [ ] Check threadgroup memory usage
  - [ ] Verify this sets up next optimization

#### Kernel 4: 1D Blocktiling
- [ ] Each thread computes multiple outputs (e.g., TM=8 elements)
  - [ ] Thread now responsible for column of 8 results
  - [ ] Reuse B values across multiple A rows
  - [ ] Cache intermediate results in registers
  - [ ] Inner loop structure: outer over dot products, inner over thread results

- [ ] Benchmark and profile
  - [ ] Measure GFLOPS (expect 2-3x speedup, ~6000-9000)
  - [ ] Check arithmetic intensity increase
  - [ ] Profile register usage

#### Kernel 5: 2D Blocktiling
- [ ] Each thread computes 2D tile (e.g., TM×TN = 8×8)
  - [ ] Allocate register array for results: `float results[TM*TN]`
  - [ ] Cache rows/columns in registers: `float regM[TM], regN[TN]`
  - [ ] Outer product pattern: `results[i*TN+j] += regM[i] * regN[j]`
  - [ ] Loop structure: BK → load registers → TM×TN outer product

- [ ] Adjust threadgroup tile sizes
  - [ ] Increase to BM=BN=128, BK=8 (or similar based on profiling)
  - [ ] Ensure enough threadgroups for occupancy

- [ ] Benchmark and profile
  - [ ] Measure GFLOPS (expect 2x speedup, ~12000-18000)
  - [ ] Should approach compute bound
  - [ ] Check occupancy and register pressure

#### Kernel 6: Vectorized Memory Access
- [ ] Vectorize global memory loads
  - [ ] Use `float4` for 128-bit vectorized loads
  - [ ] Cast pointers: `reinterpret_cast<float4*>(&A[...])`
  - [ ] Ensure alignment requirements are met

- [ ] Transpose A during threadgroup memory loading
  - [ ] Transpose A while copying from global to threadgroup memory
  - [ ] Enables vectorized threadgroup memory loads for A
  - [ ] B already has good layout

- [ ] Benchmark and profile
  - [ ] Measure GFLOPS (expect 10-20% speedup, ~15000-20000)
  - [ ] Verify vectorized loads in Metal Debugger
  - [ ] Check memory bandwidth is closer to peak

#### Kernel 7-9: Autotuning
- [ ] Parameterize kernel configuration
  - [ ] Template parameters: BM, BN, BK, TM, TN
  - [ ] Currently hardcoded values become configurable

- [ ] Create autotuning script
  - [ ] Grid search over valid configurations
  - [ ] Test: BM,BN ∈ {32, 64, 128}, BK ∈ {8, 16}, TM,TN ∈ {4, 8}
  - [ ] Ensure divisibility constraints (BM/BN divisible by TM/TN, etc.)
  - [ ] Run benchmark for each configuration

- [ ] Find optimal configuration per matrix size
  - [ ] Small matrices (512×512): may prefer different config
  - [ ] Medium matrices (2048×2048): typical use case
  - [ ] Large matrices (4096×4096): maximum throughput
  - [ ] Store best configs or implement runtime dispatch

- [ ] Benchmark and profile
  - [ ] Measure GFLOPS (expect 5-10% gain depending on size)
  - [ ] Document optimal configs for M1/M2/M3

#### Kernel 10: SIMD-group Tiling (Advanced, Optional)
- [ ] Add SIMD-group level tiling within each threadgroup
  - [ ] Calculate SIMD-group ID: `simdgroup_index_in_threadgroup`
  - [ ] Each SIMD-group computes WM×WN tile
  - [ ] Thread computes TM×TN within SIMD-group tile
  - [ ] New loop structure: BK → SIMD-group iter → thread iter

- [ ] Optimize SIMD-group memory access patterns
  - [ ] Ensure SIMD-group threads access consecutive threadgroup memory
  - [ ] May help with bank conflicts

- [ ] Benchmark and profile
  - [ ] Measure GFLOPS (expect 5-15% gain, ~20000-23000)
  - [ ] Compare against Metal Performance Shaders GEMM
  - [ ] Should be within 90-95% of MPS on large matrices

#### Testing Throughout
- [ ] Create `tests/test_gemm.py`
  - [ ] Test each kernel variant separately
  - [ ] Compare all kernels against NumPy reference
  - [ ] Test various matrix sizes and shapes
  - [ ] Test transpose cases
  - [ ] Gradcheck matmul operation

- [ ] Create benchmark suite
  - [ ] Benchmark all kernel variants
  - [ ] Plot GFLOPS vs matrix size
  - [ ] Compare against Metal Performance Shaders

#### Examples
- [ ] Update `examples/mlp.py`
  - [ ] Train on MNIST/CIFAR after Kernel 6+
  - [ ] Report training time per epoch
  - [ ] Compare CPU vs GPU speedup

**Stage 3 Milestones:** 
- ✅ After Kernel 1-3: Correctness established, understanding memory hierarchy
- ✅ After Kernel 4-5: Practical performance, can train real models
- ✅ After Kernel 6-10: Optimized performance, within 10% of MPS GEMM

---

### Stage 4: LayerNorm + GELU (~400-600 LOC)

#### LayerNorm Kernel
- [ ] Create `shaders/ops_layernorm.metal`
  - [ ] Implement two-pass forward kernel
    - [ ] Pass 1: Compute mean and variance
    - [ ] Pass 2: Normalize with `(x - mean) / sqrt(var + eps)`
  - [ ] Implement backward kernel
    - [ ] Compute gradients w.r.t. input, scale, and bias
  - [ ] Use Welford's algorithm for numerical stability

#### GELU Kernel
- [ ] Add `gelu` to `shaders/ops_elementwise.metal`
  - [ ] Implement GELU approximation: `0.5 * x * (1 + tanh(...))`
  - [ ] Implement backward pass
  - [ ] Test against PyTorch implementation

#### Neural Network Modules
- [ ] Create `core/nn/layernorm.py`
  - [ ] Implement `LayerNorm` module
  - [ ] Add learnable scale and bias parameters
  - [ ] Handle normalized_shape configuration

- [ ] Create `core/nn/gelu.py`
  - [ ] Implement `GELU` activation module

#### Testing
- [ ] Test LayerNorm correctness vs PyTorch
- [ ] Test GELU correctness vs PyTorch
- [ ] Gradcheck both operations
- [ ] Test numerical stability on extreme values

#### Example
- [ ] Create `examples/transformer_block.py`
  - [ ] Implement MLP + LayerNorm block (no attention yet)
  - [ ] Train on simple task
  - [ ] Profile memory bandwidth

#### Profiling
- [ ] Profile LayerNorm with Metal Debugger
- [ ] Identify memory bottlenecks
- [ ] Measure bandwidth utilization
- [ ] Optimize if needed

**Stage 4 Milestone:** ✅ Transformer block (MLP + LayerNorm) trains on GPU

---

### Stage 5: Attention (~1200-2000 LOC)

#### Stage 5A: Naive Attention

##### Supporting Operations
- [ ] Update `core/tensor.py`
  - [ ] Add `.reshape()` method
  - [ ] Add `.transpose()` method (materialize, no views)
  - [ ] Add `.view()` as alias for reshape

##### Embedding Layer
- [ ] Create `core/nn/embedding.py`
  - [ ] Implement `Embedding` module
  - [ ] Implement forward: 1D gather/index operation
  - [ ] Implement backward: scatter gradients

- [ ] Create `shaders/ops_embedding.metal`
  - [ ] Implement Metal embedding lookup kernel
  - [ ] Implement gradient scatter kernel

##### Attention Implementation
- [ ] Create `core/nn/attention.py`
  - [ ] Implement `MultiHeadAttention` module
  - [ ] Split into Q, K, V projections
  - [ ] Implement attention computation:
    - [ ] `scores = (Q @ K^T) / sqrt(d_k)`
    - [ ] `P = softmax(scores)`
    - [ ] `out = P @ V`
  - [ ] Add causal masking support
  - [ ] Merge heads and project output

- [ ] Implement backward pass
  - [ ] Gradient through output projection
  - [ ] Gradient through attention weights (P @ V)
  - [ ] Gradient through softmax
  - [ ] Gradient through scaled dot-product
  - [ ] Gradient through Q, K, V projections

##### GPT Architecture
- [ ] Create `examples/gpt_mini.py`
  - [ ] Implement `GPTBlock` (attention + MLP + LayerNorm)
  - [ ] Implement `GPT` model class
  - [ ] Add token and position embeddings
  - [ ] Stack multiple transformer blocks
  - [ ] Add final LayerNorm and output head

##### Training
- [ ] Implement next-token prediction loss
- [ ] Train on small text dataset (e.g., TinyShakespeare)
- [ ] Verify model can overfit small dataset
- [ ] Test generation (sample tokens autoregressively)

##### Testing
- [ ] Test attention output correctness vs PyTorch
- [ ] Gradcheck attention module
- [ ] Test with different sequence lengths
- [ ] Test causal masking

#### Stage 5B: Flash Attention (~800-1200 LOC)

Following the 6-part Flash Attention optimization guide from lubits.ch/flash, adapted for Metal on Apple Silicon M1/M2/M3.

##### Part 1-2: Foundation & Building Blocks

**Metal API Learning**
- [ ] Study Metal equivalents for GPU operations:
  - [ ] `simdgroup_matrix_storage` for 8×8 matrix tile operations
  - [ ] `async_copy` for efficient global→threadgroup transfers
  - [ ] `simdgroup_load`/`simdgroup_store` for threadgroup→SIMD-group transfers
  - [ ] Threadgroup memory (on-chip shared memory within threadgroup)
  - [ ] SIMD-groups of 32 threads (Metal's parallel execution unit)

**Fragment Operations**
- [ ] Understand 8×8 tile operations with `simdgroup_matrix_storage`
- [ ] Study thread-to-element mapping in SIMD-groups
- [ ] Learn threadgroup memory banking structure (32 banks of 4B each)

**Memory Hierarchy**
- [ ] Global memory (device buffers)
- [ ] Threadgroup memory (on-chip, shared within threadgroup)
- [ ] SIMD-group register file (private to SIMD-group)

##### Part 3: Kernel 1 - Baseline Flash Attention (~150-200 LOC)

**Block Configuration**
- [ ] Choose initial block sizes: B_r=64 (query rows), B_c=64 (key/value rows), d_head=128
- [ ] Use 4 SIMD-groups (128 threads total) per threadgroup
- [ ] Calculate threadgroup memory requirements: ~48KiB for Q/K/V/O tiles

**Work Distribution**
- [ ] Map threadgroups to (batch, head, query_block) grid
- [ ] Distribute query rows across 4 SIMD-groups (16 rows per SIMD-group)
- [ ] Implement cooperative K/V loading across all SIMD-groups

**Core Algorithm Implementation**
- [ ] Create `shaders/ops_attention.metal`
- [ ] Implement Prologue:
  - [ ] Initialize threadgroup memory pointers
  - [ ] Load Q tile: Global → Threadgroup → SIMD-group registers
  - [ ] Initialize online softmax statistics (m=-inf, l=0.0)
  - [ ] Zero output accumulator

- [ ] Implement Mainloop (iterate over K/V blocks):
  - [ ] Load K block: Global → Threadgroup → SIMD-group registers
  - [ ] Compute attention scores: S = Q @ K^T using `simdgroup_multiply_accumulate`
  - [ ] Apply online softmax:
    - [ ] Compute row max (with SIMD-group shuffle reductions)
    - [ ] Update running max and rescale previous l/O
    - [ ] Exponentiate attention scores
    - [ ] Update running sum l
  - [ ] Convert S from fp32 to fp16 (P tile)
  - [ ] Load V block: Global → Threadgroup → SIMD-group registers  
  - [ ] Accumulate output: O += P @ V

- [ ] Implement Epilogue:
  - [ ] Final softmax normalization: O /= l
  - [ ] Convert O from fp32 to fp16
  - [ ] Store O: SIMD-group registers → Threadgroup → Global

**Memory Operations**
- [ ] Implement Global ↔ Threadgroup transfers (16B vectorized, async copy)
- [ ] Implement Threadgroup → SIMD-group loads (`simdgroup_load`, 8×8 tiles)
- [ ] Implement SIMD-group → Threadgroup stores (standard 4B stores)

**Synchronization**
- [ ] Add `threadgroup_barrier()` after K/V cooperative loads
- [ ] Add `threadgroup_barrier()` between mainloop iterations
- [ ] Use `simdgroup_barrier()` for SIMD-group-level operations

**Testing & Profiling**
- [ ] Test correctness vs naive attention implementation
- [ ] Profile with Metal Debugger:
  - [ ] Check occupancy (aim for high threadgroup utilization per GPU core)
  - [ ] Measure kernel execution time
  - [ ] Verify no register spills
- [ ] Benchmark performance: expect ~40-50% of MPS reference

**Kernel 1 Target:** Working Flash Attention at 40-50% of MPS performance

##### Part 4: Kernel 2 - Bank Conflicts & Swizzling (~100-150 LOC)

**Understanding Bank Conflicts**
- [ ] Profile Kernel 1 with Metal Performance Counter tool
- [ ] Identify threadgroup memory bank conflicts in:
  - [ ] Threadgroup → SIMD-group loads (ldmatrix equivalent)
  - [ ] SIMD-group → Threadgroup stores

**Bank Conflict Analysis**
- [ ] Understand 16B vectorized banking: 8 banks of 16B each (for 16B accesses)
- [ ] Recognize 8-way conflicts in naive layout
- [ ] Analyze phase-based execution (8 threads per phase)

**Swizzling Implementation**
- [ ] Create `get_swizzled_col()` function using XOR pattern: `(row ^ col)`
- [ ] Apply swizzling to all threadgroup memory accesses:
  - [ ] Global → Threadgroup copies
  - [ ] Threadgroup → SIMD-group loads
  - [ ] SIMD-group → Threadgroup stores
  - [ ] Threadgroup → Global copies

**Code Changes**
- [ ] Add swizzling to `copy_block_global_to_threadgroup()`
- [ ] Add swizzling to `copy_threadgroup_to_simdgroup()` for Q/K/V
- [ ] Add swizzling to `copy_threadgroup_to_simdgroup_transposed()` for K^T
- [ ] Add swizzling to `copy_simdgroup_to_threadgroup()` for O

**Testing & Profiling**
- [ ] Verify bank conflicts eliminated (check with profiler)
- [ ] Test correctness (results should match Kernel 1 exactly)
- [ ] Measure threadgroup memory bandwidth improvement (expect 8x reduction in wavefronts)
- [ ] Benchmark performance: expect ~80-100% of MPS reference (2x improvement)

**Kernel 2 Target:** 2x speedup, ~80-100% of MPS reference performance

##### Part 5: Kernels 3-5 - GEMM Optimizations (~250-350 LOC)

**Kernel 3: Double Buffering Global → Threadgroup (~80-100 LOC)**
- [ ] Allocate extra threadgroup memory slice for K/V (double buffering)
- [ ] Load next K/V block while computing on current block:
  - [ ] Move K load to after S computation (S = Q @ K^T)
  - [ ] Move V load to after softmax computation
  - [ ] Add special case for initial K load in prologue
- [ ] Update threadgroup memory pointers to alternate between buffers
- [ ] Add proper synchronization barriers
- [ ] Profile: expect ~93% reduction in global memory stalls
- [ ] Benchmark: expect ~99-100% of MPS reference

**Kernel 4: Fragment Interleaving (~100-150 LOC)**
- [ ] Break GEMM operations into sub-tiles along k-dimension:
  - [ ] Load 2 fragments at a time (16 elements wide in k-dimension)
  - [ ] Perform GEMM on those fragments
  - [ ] Load next 2 fragments while computing
- [ ] Reduce register pressure significantly:
  - [ ] Change storage from full tiles to 2-fragment sub-tiles
  - [ ] Enable larger block configurations
- [ ] Optimize fragment reuse:
  - [ ] Load A and B along k-dimension (not row-by-row)
  - [ ] Fragment intensity: 1.6x vs 0.89x for naive approach
- [ ] Test register usage across configurations:
  - [ ] Verify (128,32,4) no longer spills
  - [ ] Measure spill reduction for (128,64,4)
- [ ] Benchmark: expect ~100% of MPS reference, reduced latency

**Kernel 5: Double Buffering Threadgroup → SIMD-group (~70-100 LOC)**
- [ ] Allocate double register storage for K/V fragments
- [ ] Alternate between register buffers:
  - [ ] Load next fragment set before computing on current set
  - [ ] Toggle buffer index each iteration
- [ ] Add initial fragment load before GEMM loop
- [ ] Profile: slight increase in `mio_throttle` stalls expected
- [ ] Note: May show regression for some configs but enables better auto-tuning
- [ ] Benchmark: expect ~99-100% of MPS reference

**Testing for Kernels 3-5**
- [ ] Verify correctness after each kernel
- [ ] Profile memory stall reductions
- [ ] Measure register usage and spills
- [ ] Test multiple block configurations

##### Part 6: Kernel 6 - FP Instruction Fusion & Auto-Tuning (~150-200 LOC)

**FP Instruction Fusion**
- [ ] Analyze softmax instruction count per K/V tile
- [ ] Fuse scaling and exponential operations:
  - [ ] Replace separate `S *= scale` with fused multiply-add in exponent
  - [ ] Compute `m_scaled = m * scale` once per row
  - [ ] Use FMA: `S = exp2f(S * scale - m_scaled)`
- [ ] Make fast exponential explicit:
  - [ ] Use identity: `exp(x) = exp2(x * log2(e))`
  - [ ] Fold `scale * log2(e)` into single coefficient
- [ ] Update code:
  - [ ] Modify `scale_l_O()` to use `exp2f((m_cur - m_next) * scale)`
  - [ ] Modify `exponentiate_tensor()` to fuse scale and exponential
  - [ ] Remove separate `scale_S_accum()` call
- [ ] Measure instruction reduction: expect 11% fewer FP instructions
- [ ] Benchmark: expect ~99.9-100% of MPS reference

**Auto-Tuning Configuration Space**
- [ ] Define tunable parameters:
  - [ ] `d_head` ∈ {128} (fixed for this project)
  - [ ] `B_r` ∈ {64, 128} (query block rows)
  - [ ] `B_c` ∈ {32, 64} (key/value block rows)
  - [ ] `n_simdgroups` ∈ {4} (threads per threadgroup = 128)
  - [ ] `Q_fragments_persisted` ∈ {true, false}
  - [ ] `K_fragment_width` ∈ {0, 2} (0 = full tile, 2 = sub-tiles)
  - [ ] `V_fragment_width` ∈ {0, 2}
  - [ ] `double_buffer_smem_to_rf` ∈ {true, false}
  - [ ] `optimized_softmax` ∈ {true, false}

- [ ] Filter out non-viable configurations:
  - [ ] Remove configs with excessive register spills (>100B per thread)
  - [ ] Remove configs with inadequate threadgroup memory

- [ ] Create benchmarking harness:
  - [ ] Generate all valid kernel configurations
  - [ ] Compile each configuration
  - [ ] Run on representative workloads (seq_len ∈ {512, 1024, 2048, 4096})
  - [ ] Measure throughput (TFLOPs/s) and memory usage

- [ ] Analyze results:
  - [ ] Compare register usage across configs
  - [ ] Compare occupancy (threadgroup utilization per GPU core)
  - [ ] Identify sweet spots for different sequence lengths
  - [ ] Document optimal configuration for M1/M2/M3

**Testing & Validation**
- [ ] Verify numerical accuracy vs Kernel 1 baseline
- [ ] Test on various sequence lengths: 512, 1024, 2048, 4096, 8192
- [ ] Test on different head dimensions if extended
- [ ] Verify auto-tuning selects good configurations

**Final Profiling**
- [ ] Profile optimal configuration with Metal Debugger
- [ ] Analyze occupancy, register pressure, memory bandwidth
- [ ] Compare against MPS reference implementation
- [ ] Document performance characteristics:
  - [ ] Peak TFLOPs/s achieved
  - [ ] Memory bandwidth utilization
  - [ ] Sequence length scaling behavior

**Kernel 6 Target:** 100-105% of MPS reference performance on M1/M2/M3

##### Integration with Attention Module

- [ ] Update `core/nn/attention.py`
  - [ ] Add `use_flash_attention` flag to `MultiHeadAttention`
  - [ ] Add Metal kernel selection based on auto-tuned configuration
  - [ ] Dispatch to Flash Attention kernel when enabled
  - [ ] Keep naive implementation as fallback

##### Benchmarking & Comparison

- [ ] Create `benchmarks/flash_attention_bench.py`
  - [ ] Compare memory usage: naive vs flash (expect O(N²) → O(N) reduction)
  - [ ] Compare speed across sequence lengths: 512, 1024, 2048, 4096
  - [ ] Measure maximum sequence length that fits in memory
  - [ ] Compare against MPS reference implementation
  - [ ] Generate performance plots

- [ ] Update `examples/gpt_mini.py`
  - [ ] Enable Flash Attention by default
  - [ ] Report training speedup
  - [ ] Test on longer sequences (2K-4K tokens)

**Stage 5B Milestone:** ✅ Flash Attention implementation achieving 100-105% of MPS reference, enabling efficient training on 4K+ sequence lengths

**Stage 5 Milestone:** ✅ GPT-mini trains end-to-end and generates text

---

### Final Polish & Documentation

#### Code Quality
- [ ] Add docstrings to all public APIs
- [ ] Add type hints throughout codebase
- [ ] Format code consistently (black, clang-format)
- [ ] Add `__repr__` methods to key classes

#### Testing Suite
- [ ] Achieve good test coverage (>80%)
- [ ] Add integration tests for full training loops
- [ ] Add benchmark suite
- [ ] Add performance regression tests

#### Documentation
- [ ] Write README.md with installation instructions
- [ ] Document Metal kernel implementations
- [ ] Add tutorial notebooks (Jupyter)
- [ ] Document performance characteristics
- [ ] Add API reference documentation

#### Examples & Benchmarks
- [ ] Clean up all example scripts
- [ ] Add timing and memory usage reporting
- [ ] Create benchmark comparison vs PyTorch
- [ ] Document expected performance on target GPU

#### Repository Organization
- [ ] Add LICENSE file
- [ ] Add .gitignore
- [ ] Add requirements.txt / environment.yml
- [ ] Add CI/CD configuration
- [ ] Tag version releases