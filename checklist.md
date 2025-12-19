
## Complete Project Checklist

### Stage 0: Pure Python CPU Skeleton (~300-400 LOC)

#### Core Autograd System
- [x] Create `frontend/autograd.py`
  - [x] Implement `Function` base class with `forward()` and `backward()` abstract methods
  - [x] Implement `Context` class for saving tensors/data needed in backward pass
  - [x] Implement backward tape/graph structure
  - [x] Add gradient accumulation logic

- [x] Create `frontend/tensor.py`
  - [x] Implement `Tensor` class wrapping NumPy arrays
  - [x] Add `.requires_grad` flag
  - [x] Add `.grad` attribute for storing gradients
  - [x] Implement `.backward()` method that walks the computation tape
  - [x] Add `.item()` for scalar extraction
  - [x] Add `.numpy()` for NumPy conversion
  - [x] Add `.shape`, `.dtype` properties
  - [x] Implement basic tensor creation: `zeros`, `ones`, `randn`, `arange`

#### Functional Operations
- [x] Create `frontend/functional.py`
  - [x] Implement `Add` function (forward/backward)
  - [x] Implement `Mul` function (forward/backward)
  - [x] Implement `MatMul` function (forward/backward)
  - [x] Implement `Sum` function (forward/backward)
  - [x] Implement `ReLU` function (forward/backward)
  - [x] Implement `CrossEntropy` function (forward/backward)
  - [x] Add convenience wrappers: `add()`, `mul()`, `matmul()`, etc.

#### Neural Network Primitives
- [x] Create `frontend/nn/module.py`
  - [x] Implement `Module` base class
  - [x] Add `__call__()` that invokes `forward()`
  - [x] Implement `.parameters()` method that recursively finds all Parameters
  - [x] Implement `.to(device)` method
  - [x] Implement `Parameter` class (Tensor with `requires_grad=True`)

- [x] Create `frontend/nn/linear.py`
  - [x] Implement `Linear` layer with weight initialization
  - [x] Add optional bias parameter
  - [x] Implement forward pass: `y = x @ W^T + b`

#### Optimizer
- [x] Create `frontend/optim/sgd.py`
  - [x] Implement `SGD` optimizer (required: implement at least one optimizer)
  - [x] Add `.zero_grad()` method
  - [x] Add `.step()` method with learning rate
  - [x] Add momentum support

#### Testing & Validation
- [x] Create `tests/test_ops.py`
  - [x] Test basic tensor operations
  - [x] Test autograd correctness
  - [x] Implement numerical gradient checking

- [x] Create `tests/mlp.py`
  - [x] Implement simple MLP class
  - [x] Train on XOR and tiny MNIST
  - [x] Verify gradients are correct

**Stage 0 Milestone:** ✅ MLP trains on CPU with NumPy backend

---

### Stage 1: GPU Memory + Elementwise Kernels (~400-600 LOC)

#### RunPod Setup
- [ ] Create RunPod account at runpod.io
- [ ] Launch GPU pod (RTX 3080 Ti recommended)
  - [ ] Choose PyTorch template (includes CUDA Toolkit, nvcc, Python)
  - [ ] Note SSH connection details: `ssh root@<pod-ip> -p <port>`
  - [ ] Set up persistent volume for data (optional)
- [ ] SSH into pod and verify setup
  - [ ] Check CUDA: `nvcc --version`
  - [ ] Check GPU: `nvidia-smi`
  - [ ] Install cupy: `pip install cupy-cuda12x` (adjust for CUDA version)
  - [ ] Test cupy: `python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())"`
- [ ] Set up sync workflow
  - [ ] Use VS Code Remote-SSH for seamless development
- [ ] Create project directory on pod: `mkdir -p /workspace/FlamingCamel`

#### Kernel Development System
- [ ] Create `backend/ops_elementwise.cu` with CUDA kernels
- [ ] Create `scripts/compile_kernels.sh` on RunPod:
  - [ ] Compile all `.cu` files to `.ptx`
  - [ ] Example: `nvcc -ptx backend/ops_elementwise.cu -o backend/ops_elementwise.ptx`
- [ ] Test kernel compilation directly on pod

#### Backend Infrastructure (Direct CUDA)
- [ ] Create `frontend/backend.py`
  - [ ] Initialize cupy and detect GPU
  - [ ] Load compiled `.ptx` kernels using `cupy.RawKernel`
  - [ ] Create kernel wrapper functions
  - [ ] Add error checking for CUDA errors
  - [ ] Cache loaded kernels to avoid reloading

#### Tensor Device Support
- [ ] Update `frontend/tensor.py`
  - [ ] Store data as `self._data`: `cupy.ndarray` for GPU or `numpy.ndarray` for CPU
  - [ ] Add `.device` property: return "cuda" or "cpu" based on array type
  - [ ] Implement `.to("cuda")` method: convert to cupy array
  - [ ] Implement `.to("cpu")` method: convert to numpy array
  - [ ] Ensure `.grad` tensors stay on same device as parent tensor

#### Elementwise Kernels
- [ ] Create `backend/ops_elementwise.cu`
  - [ ] Implement `add_kernel`: `__global__ void add_kernel(float* a, float* b, float* c, int n)`
  - [ ] Implement `mul_kernel`, `relu_kernel`, `relu_backward_kernel`
  - [ ] Each kernel: 1 thread per element, grid-stride loop
- [ ] Compile kernels on RunPod:
  - [ ] `nvcc -ptx backend/ops_elementwise.cu -o backend/ops_elementwise.ptx`
  - [ ] Verify compilation succeeds
- [ ] Update `frontend/backend.py` with kernel loading:
  - [ ] Load `.ptx` files with `cupy.RawKernel`
  - [ ] Create wrapper functions: `cuda_add()`, `cuda_mul()`, `cuda_relu()`
  - [ ] Handle grid/block dimensions automatically

- [ ] Update `frontend/functional.py`
  - [ ] Add device dispatch: if device="cuda", call CUDA kernels
  - [ ] If device="cpu", use NumPy operations
  - [ ] Handle both forward and backward passes

#### Testing
- [ ] Create `tests/test_gpu_ops.py`
  - [ ] Test CPU vs CUDA correctness for `add`, `mul`, `relu`
  - [ ] Test different tensor shapes
  - [ ] Test gradient correctness on CUDA

- [ ] Update `examples/mlp.py`
  - [ ] Add `.to("cuda")` to model and data
  - [ ] Verify training works on GPU

#### Profiling
- [ ] Profile kernel dispatch overhead with Nsight Systems
- [ ] Measure memory transfer times
- [ ] Document performance baseline

**Stage 1 Milestone:** ✅ MLP trains on GPU with custom CUDA kernels

---

### Stage 2: Reductions + Softmax (~500-700 LOC)

#### Reduction Kernels
- [ ] Create `backend/ops_reduce.cu`
  - [ ] Implement block-wise `sum` reduction kernel
  - [ ] Use shared memory for per-block partial sums
  - [ ] Implement second kernel for final reduction across blocks
  - [ ] Implement `max` reduction (similar structure)
  - [ ] Use warp shuffle primitives (`__shfl_down_sync`)

#### Softmax Kernels
- [ ] Create `backend/ops_softmax.cu`
  - [ ] Implement `softmax` forward kernel (max-subtract for stability)
  - [ ] Each block handles one row, use shared memory for max/sum
  - [ ] Implement `log_softmax` forward kernel
  - [ ] Implement `softmax` backward kernel
  - [ ] Implement `log_softmax` backward kernel

#### Functional API
- [ ] Update `frontend/functional.py`
  - [ ] Add `sum()` operation with CUDA dispatch
  - [ ] Add `max()` operation with CUDA dispatch
  - [ ] Add `softmax()` using CUDA kernel
  - [ ] Add `log_softmax()` using CUDA kernel
  - [ ] Add stable `cross_entropy()` using log_softmax

#### Testing
- [ ] Test reduction correctness (CPU vs CUDA)
- [ ] Test softmax numerical stability
  - [ ] Test cross_entropy gradients
- [ ] Gradcheck all new operations

#### Example
- [ ] Create `examples/classifier.py`
  - [ ] Train classifier with softmax + cross-entropy loss
  - [ ] Verify stable training

#### Profiling
- [ ] Profile reduction kernel with Nsight Compute
- [ ] Measure shared memory usage
- [ ] Analyze warp divergence
- [ ] Optimize block size

**Stage 2 Milestone:** ✅ Classifier with stable softmax+loss on GPU

---

### Stage 3: Matrix Multiplication (~1000-1500 LOC)

Following exactly https://siboehm.com/articles/22/CUDA-MMM

#### Kernel 1: Naive Implementation
- [ ] Create `backend/ops_matmul.cu`
  - [ ] Implement naive GEMM: each thread computes one output element
  - [ ] Triple nested loop: `C[row][col] = sum(A[row][k] * B[k][col])`
  - [ ] Grid configuration: `dim3 threads(16, 16); dim3 blocks((N+15)/16, (M+15)/16)`
  - [ ] Test correctness vs NumPy on small matrices
  - [ ] Implement backward pass: `dW = X^T @ dY`, `dX = dY @ W^T`

- [ ] Update `frontend/nn/linear.py`
  - [ ] Use CUDA matmul when on cuda
  - [ ] Verify backward pass correctness

- [ ] Benchmark and profile
  - [ ] Measure GFLOPS (expect ~100-300)
  - [ ] Check memory bandwidth utilization (should be very low)

#### Kernel 2: Global Memory Coalescing
- [ ] Remap thread indexing for coalesced access
  - [ ] Change from 2D `thread_position_in_threadgroup` to 1D
  - [ ] Ensure consecutive threads access consecutive memory
  - [ ] Threads in same warp load contiguous A elements

- [ ] Benchmark and profile
  - [ ] Measure GFLOPS (expect 6-8x speedup)
  - [ ] Verify improved memory bandwidth (should increase significantly)

#### Kernel 3: Shared Memory Cache-Blocking
- [ ] Implement shared memory caching
  - [ ] Allocate shared memory for A and B tiles (e.g., 32×32 each)
  - [ ] Outer loop: advance through K dimension
  - [ ] Load tiles cooperatively from global to shared memory
  - [ ] Use `__syncthreads()` before and after computation
  - [ ] Inner loop: compute partial dot products using shared memory data

- [ ] Benchmark and profile
  - [ ] Measure GFLOPS (expect small improvement, ~2000-3000)
  - [ ] Check shared memory usage
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

- [ ] Adjust block tile sizes
  - [ ] Increase to BM=BN=128, BK=8 (or similar based on profiling)
  - [ ] Ensure enough blocks for occupancy

- [ ] Benchmark and profile
  - [ ] Measure GFLOPS (expect 2x speedup, ~12000-18000)
  - [ ] Should approach compute bound
  - [ ] Check occupancy and register pressure

#### Kernel 6: Vectorized Memory Access
- [ ] Vectorize global memory loads
  - [ ] Use `float4` for 128-bit vectorized loads
  - [ ] Cast pointers: `reinterpret_cast<float4*>(&A[...])`
  - [ ] Ensure alignment requirements are met

- [ ] Transpose A during shared memory loading
  - [ ] Transpose A while copying from global to shared memory
  - [ ] Enables vectorized shared memory loads for A
  - [ ] B already has good layout

- [ ] Benchmark and profile
  - [ ] Measure GFLOPS (expect 10-20% speedup, ~15000-20000)
  - [ ] Verify vectorized loads in Nsight Compute
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
  - [ ] Document optimal configs for target GPU

#### Kernel 10: Warp Tiling
- [ ] Add warp level tiling within each block
  - [ ] Calculate warp ID: `threadIdx.x / 32`
  - [ ] Each warp computes WM×WN tile
  - [ ] Thread computes TM×TN within warp tile
  - [ ] New loop structure: BK → warp iter → thread iter

- [ ] Optimize warp memory access patterns
  - [ ] Ensure warp threads access consecutive shared memory
  - [ ] May help with bank conflicts

- [ ] Benchmark and profile
  - [ ] Measure GFLOPS (expect 5-15% gain, ~20000-23000)
  - [ ] Compare against cuBLAS GEMM
  - [ ] Should be within 90-95% of cuBLAS on large matrices

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
  - [ ] Compare against cuBLAS

#### Examples
- [ ] Update `examples/mlp.py`
  - [ ] Train on MNIST/CIFAR after Kernel 6+
  - [ ] Report training time per epoch
  - [ ] Compare CPU vs GPU speedup

**Stage 3 Milestones:** 
- ✅ After Kernel 1-3: Correctness established, understanding memory hierarchy
- ✅ After Kernel 4-5: Practical performance, can train real models
- ✅ After Kernel 6-10: Optimized performance, within 10% of cuBLAS GEMM

---

### Stage 4: LayerNorm + GELU + Adam (~500-700 LOC)

#### Neural Network Modules
- [ ] Create `frontend/nn/layernorm.py`
  - [ ] Implement `LayerNorm` module
  - [ ] Add learnable scale and bias parameters
  - [ ] Handle normalized_shape configuration

- [ ] Create `frontend/nn/gelu.py`
  - [ ] Implement `GELU` activation module

  #### Adam Optimizer
- [ ] Create `frontend/optim/adam.py`
  - [ ] Implement `Adam` optimizer
  - [ ] Add first and second moment estimates
  - [ ] Implement bias correction
  - [ ] Add `.zero_grad()` and `.step()` methods

#### LayerNorm Kernel
- [ ] Create `backend/ops_layernorm.cu`
  - [ ] Implement two-pass forward kernel
    - [ ] Pass 1: Compute mean and variance
    - [ ] Pass 2: Normalize with `(x - mean) / sqrt(var + eps)`
  - [ ] Implement backward kernel
    - [ ] Compute gradients w.r.t. input, scale, and bias
  - [ ] Use Welford's algorithm for numerical stability

#### GELU Kernel
- [ ] Add `gelu` to `backend/ops_elementwise.cu`
  - [ ] Implement GELU approximation: `0.5 * x * (1 + tanh(...))`
  - [ ] Implement backward pass
  - [ ] Test against PyTorch implementation

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
- [ ] Profile LayerNorm with Nsight Compute
- [ ] Identify memory bottlenecks
- [ ] Measure bandwidth utilization
- [ ] Optimize if needed

**Stage 4 Milestone:** ✅ Transformer block (MLP + LayerNorm) trains on GPU

---

### Stage 5: Attention (~1200-2000 LOC)

#### Stage 5A: Naive Attention

##### Supporting Operations
- [ ] Update `frontend/tensor.py`
  - [ ] Add `.reshape()` method
  - [ ] Add `.transpose()` method (materialize, no views)
  - [ ] Add `.view()` as alias for reshape

##### Embedding Layer
- [ ] Create `frontend/nn/embedding.py`
  - [ ] Implement `Embedding` module
  - [ ] Implement forward: 1D gather/index operation
  - [ ] Implement backward: scatter gradients

- [ ] Create `backend/ops_embedding.cu`
  - [ ] Implement CUDA embedding lookup kernel
  - [ ] Implement gradient scatter kernel

##### Attention Implementation
- [ ] Create `frontend/nn/attention.py`
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

Following the Flash Attention optimization guide from lubits.ch/flash (a 10-part series), we'll implement through Part 6, which covers 6 kernel iterations. Parts 7-10 cover additional profiling and optimizations.

##### Part 1-2: Foundation & Building Blocks (Preparation, no kernel implementation)

**CUDA API Learning**
- [ ] Study CUDA equivalents for GPU operations:
  - [ ] `wmma` (Warp Matrix Multiply-Accumulate) for 16×16 matrix tile operations
  - [ ] `cp.async` for efficient global→shared memory transfers
  - [ ] `ldmatrix`/`stmatrix` for shared memory→warp register transfers
  - [ ] Shared memory (on-chip shared memory within block)
  - [ ] Warps of 32 threads (CUDA's parallel execution unit)

**Fragment Operations**
- [ ] Understand 16×16 tile operations with `wmma::fragment`
- [ ] Study thread-to-element mapping in warps
- [ ] Learn shared memory banking structure (32 banks of 4B each)

**Memory Hierarchy**
- [ ] Global memory (device buffers)
- [ ] Shared memory (on-chip, shared within block)
- [ ] Warp register file (private to warp)

##### Part 3: Kernel 1 - Baseline Flash Attention (~150-200 LOC)

**Block Configuration**
- [ ] Choose initial block sizes: B_r=64 (query rows), B_c=64 (key/value rows), d_head=128
- [ ] Use 4 warps (128 threads total) per block
- [ ] Calculate shared memory requirements: ~48KiB for Q/K/V/O tiles

**Work Distribution**
- [ ] Map blocks to (batch, head, query_block) grid
- [ ] Distribute query rows across 4 warps (16 rows per warp)
  - [ ] Implement cooperative K/V loading across all warps

**Core Algorithm Implementation**
- [ ] Create `backend/ops_attention.cu`
- [ ] Implement Prologue:
  - [ ] Initialize shared memory pointers
  - [ ] Load Q tile: Global → Shared → Warp registers
  - [ ] Initialize online softmax statistics (m=-inf, l=0.0)
  - [ ] Zero output accumulator

- [ ] Implement Mainloop (iterate over K/V blocks):
  - [ ] Load K block: Global → Shared → Warp registers
  - [ ] Compute attention scores: S = Q @ K^T using `wmma::mma_sync`
  - [ ] Apply online softmax:
    - [ ] Compute row max (with warp shuffle reductions)
    - [ ] Update running max and rescale previous l/O
    - [ ] Exponentiate attention scores
    - [ ] Update running sum l
  - [ ] Convert S from fp32 to fp16 (P tile)
  - [ ] Load V block: Global → Shared → Warp registers  
  - [ ] Accumulate output: O += P @ V

- [ ] Implement Epilogue:
  - [ ] Final softmax normalization: O /= l
  - [ ] Convert O from fp32 to fp16
  - [ ] Store O: Warp registers → Shared → Global

**Memory Operations**
- [ ] Implement Global ↔ Shared transfers (16B vectorized, cp.async)
- [ ] Implement Shared → Warp loads (`ldmatrix`, 16×16 tiles)
- [ ] Implement Warp → Shared stores (standard 4B stores)

**Synchronization**
- [ ] Add `__syncthreads()` after K/V cooperative loads
- [ ] Add `__syncthreads()` between mainloop iterations
- [ ] Use `__syncwarp()` for warp-level operations

**Testing & Profiling**
- [ ] Test correctness vs naive attention implementation
- [ ] Profile with Nsight Compute:
  - [ ] Check occupancy (aim for high block utilization per SM)
  - [ ] Measure kernel execution time
  - [ ] Verify no register spills
- [ ] Benchmark performance: expect ~40-50% of cuBLAS reference

**Kernel 1 Target:** Working Flash Attention at 40-50% of cuBLAS performance

##### Part 4: Kernel 2 - Bank Conflicts & Swizzling (~100-150 LOC)

**Understanding Bank Conflicts**
- [ ] Profile Kernel 1 with Nsight Compute
- [ ] Identify shared memory bank conflicts in:
  - [ ] Shared → Warp loads (ldmatrix)
  - [ ] Warp → Shared stores

**Bank Conflict Analysis**
- [ ] Understand 16B vectorized banking: 8 banks of 16B each (for 16B accesses)
- [ ] Recognize 8-way conflicts in naive layout
- [ ] Analyze phase-based execution (8 threads per phase)

**Swizzling Implementation**
- [ ] Create `get_swizzled_col()` function using XOR pattern: `(row ^ col)`
- [ ] Apply swizzling to all shared memory accesses:
  - [ ] Global → Shared copies
  - [ ] Shared → Warp loads
  - [ ] Warp → Shared stores
  - [ ] Shared → Global copies

**Code Changes**
- [ ] Add swizzling to `copy_block_global_to_shared()`
- [ ] Add swizzling to `copy_shared_to_warp()` for Q/K/V
- [ ] Add swizzling to `copy_shared_to_warp_transposed()` for K^T
- [ ] Add swizzling to `copy_warp_to_shared()` for O

**Testing & Profiling**
- [ ] Verify bank conflicts eliminated (check with profiler)
- [ ] Test correctness (results should match Kernel 1 exactly)
- [ ] Measure shared memory bandwidth improvement (expect 8x reduction in wavefronts)
- [ ] Benchmark performance: expect ~80-100% of cuBLAS reference (2x improvement)

**Kernel 2 Target:** 2x speedup, ~80-100% of cuBLAS reference performance

##### Part 5: Kernels 3-5 - GEMM Optimizations (~250-350 LOC)

**Kernel 3: Double Buffering Global → Shared (~80-100 LOC)**
- [ ] Allocate extra shared memory slice for K/V (double buffering)
- [ ] Load next K/V block while computing on current block:
  - [ ] Move K load to after S computation (S = Q @ K^T)
  - [ ] Move V load to after softmax computation
  - [ ] Add special case for initial K load in prologue
- [ ] Update shared memory pointers to alternate between buffers
- [ ] Add proper synchronization barriers
- [ ] Profile: expect ~93% reduction in global memory stalls
- [ ] Benchmark: expect ~99-100% of cuBLAS reference

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
- [ ] Benchmark: expect ~100% of cuBLAS reference, reduced latency

**Kernel 5: Double Buffering Shared → Warp (~70-100 LOC)**
- [ ] Allocate double register storage for K/V fragments
- [ ] Alternate between register buffers:
  - [ ] Load next fragment set before computing on current set
  - [ ] Toggle buffer index each iteration
- [ ] Add initial fragment load before GEMM loop
- [ ] Profile: slight increase in memory stalls expected
- [ ] Note: May show regression for some configs but enables better auto-tuning
- [ ] Benchmark: expect ~99-100% of cuBLAS reference

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
- [ ] Benchmark: expect ~99.9-100% of cuBLAS reference

**Auto-Tuning Configuration Space**
- [ ] Define tunable parameters:
  - [ ] `d_head` ∈ {128} (fixed for this project)
  - [ ] `B_r` ∈ {64, 128} (query block rows)
  - [ ] `B_c` ∈ {32, 64} (key/value block rows)
  - [ ] `n_warps` ∈ {4} (threads per block = 128)
  - [ ] `Q_fragments_persisted` ∈ {true, false}
  - [ ] `K_fragment_width` ∈ {0, 2} (0 = full tile, 2 = sub-tiles)
  - [ ] `V_fragment_width` ∈ {0, 2}
  - [ ] `double_buffer_smem_to_rf` ∈ {true, false}
  - [ ] `optimized_softmax` ∈ {true, false}

- [ ] Filter out non-viable configurations:
  - [ ] Remove configs with excessive register spills (>100B per thread)
  - [ ] Remove configs with inadequate shared memory

- [ ] Create benchmarking harness:
  - [ ] Generate all valid kernel configurations
  - [ ] Compile each configuration
  - [ ] Run on representative workloads (seq_len ∈ {512, 1024, 2048, 4096})
  - [ ] Measure throughput (TFLOPs/s) and memory usage

- [ ] Analyze results:
  - [ ] Compare register usage across configs
  - [ ] Compare occupancy (block utilization per SM)
  - [ ] Identify sweet spots for different sequence lengths
  - [ ] Document optimal configuration for target GPU

**Testing & Validation**
- [ ] Verify numerical accuracy vs Kernel 1 baseline
- [ ] Test on various sequence lengths: 512, 1024, 2048, 4096, 8192
- [ ] Test on different head dimensions if extended
- [ ] Verify auto-tuning selects good configurations

**Final Profiling**
- [ ] Profile optimal configuration with Nsight Compute
- [ ] Analyze occupancy, register pressure, memory bandwidth
- [ ] Compare against cuBLAS reference implementation
- [ ] Document performance characteristics:
  - [ ] Peak TFLOPs/s achieved
  - [ ] Memory bandwidth utilization
  - [ ] Sequence length scaling behavior

**Kernel 6 Target:** 100-105% of cuBLAS reference performance on target GPU

**Note:** The lubits.ch/flash series continues with Parts 7-10, covering additional profiling, instruction reduction, and final optimizations. This project scope ends at Part 6 (6 kernels total), which achieves near-optimal performance.

##### Integration with Attention Module

- [ ] Update `frontend/nn/attention.py`
  - [ ] Add `use_flash_attention` flag to `MultiHeadAttention`
  - [ ] Add CUDA kernel selection based on auto-tuned configuration
  - [ ] Dispatch to Flash Attention kernel when enabled
  - [ ] Keep naive implementation as fallback

##### Benchmarking & Comparison

- [ ] Create `benchmarks/flash_attention_bench.py`
  - [ ] Compare memory usage: naive vs flash (expect O(N²) → O(N) reduction)
  - [ ] Compare speed across sequence lengths: 512, 1024, 2048, 4096
  - [ ] Measure maximum sequence length that fits in memory
  - [ ] Compare against cuBLAS reference implementation
  - [ ] Generate performance plots

- [ ] Update `examples/gpt_mini.py`
  - [ ] Enable Flash Attention by default
  - [ ] Report training speedup
  - [ ] Test on longer sequences (2K-4K tokens)

**Stage 5B Milestone:** ✅ Flash Attention implementation achieving 100-105% of cuBLAS reference, enabling efficient training on 4K+ sequence lengths

**Stage 5 Milestone:** ✅ GPT-mini trains end-to-end and generates text

---

### Stage 6: CNN Layers - Convolution + Pooling (~1500-2000 LOC)

Following the optimization guide from https://www.evl.uic.edu/sjames/cs525/final.html

#### Stage 6A: Naive Implementation & Basic Layers

##### Supporting Operations
- [ ] Update `frontend/tensor.py`
  - [ ] Ensure `.reshape()` method exists (may already be done in Stage 5)
  - [ ] Ensure `.transpose()` method exists (may already be done in Stage 5)
  - [ ] Add `.pad()` method for zero-padding boundaries

##### Conv2d Layer
- [ ] Create `frontend/nn/conv2d.py`
  - [ ] Implement `Conv2d` module
  - [ ] Add learnable weight kernel: shape (out_channels, in_channels, kernel_h, kernel_w)
  - [ ] Add optional bias parameter: shape (out_channels,)
  - [ ] Support parameters: kernel_size, stride, padding, in_channels, out_channels
  - [ ] Implement forward pass (im2col + matmul approach for baseline)
  - [ ] Implement backward pass: gradients w.r.t. input, weight, and bias

##### Pooling Layers
- [ ] Create `frontend/nn/pooling.py`
  - [ ] Implement `MaxPool2d` module
  - [ ] Support configurable kernel_size and stride
  - [ ] Forward: sliding window maximum
  - [ ] Backward: route gradients to max positions only
  - [ ] Store max indices for backward pass

##### Step 0: Naive CUDA Convolution (~150-200 LOC)
- [ ] Create `backend/ops_conv2d.cu`
  - [ ] Implement naive 2D convolution kernel
  - [ ] Each thread computes one output pixel
  - [ ] Nested loops over kernel height and width
  - [ ] Global memory access for every kernel element
  - [ ] Handle boundary conditions with zero-padding
  - [ ] Implement forward kernel
  - [ ] Implement backward kernels (gradient w.r.t. input and weights)

- [ ] Update `frontend/backend.py`
  - [ ] Load and initialize conv2d kernels
  - [ ] Add grid/block dimension calculation for 2D operations

- [ ] Update `frontend/functional.py`
  - [ ] Add `conv2d()` operation with device dispatch
  - [ ] Handle CPU (NumPy) and CUDA paths

##### Testing & Validation (6A)
- [ ] Create `tests/test_conv.py`
  - [ ] Test Conv2d correctness against PyTorch
  - [ ] Test different kernel sizes: 3×3, 5×5, 7×7
  - [ ] Test different stride and padding configurations
  - [ ] Test various channel counts
  - [ ] Gradcheck conv2d operation

- [ ] Create `examples/lenet.py`
  - [ ] Implement LeNet-5 architecture
    - [ ] Conv(1→6, 5×5) → ReLU → MaxPool(2×2)
    - [ ] Conv(6→16, 5×5) → ReLU → MaxPool(2×2)
    - [ ] Flatten → Linear(400→120) → ReLU
    - [ ] Linear(120→84) → ReLU → Linear(84→10)
  - [ ] Train on MNIST dataset
  - [ ] Verify model converges (slow training is expected at this stage)

**Stage 6A Milestone:** ✅ LeNet trains on MNIST with naive CUDA kernels (slow but correct)

#### Stage 6B: Optimized CUDA Convolution Kernels

##### Step 1: Shared Memory Caching (~200-250 LOC)
- [ ] Modify convolution kernel in `backend/ops_conv2d.cu`
  - [ ] Allocate shared memory for input tiles (e.g., 32×32)
  - [ ] Set block size to 16×16, shared memory to 32×32
  - [ ] Each thread loads 4 values from global memory
  - [ ] Include apron regions (kernel radius padding) in shared memory
  - [ ] Add `__syncthreads()` after loading to shared memory
  - [ ] Compute convolution from shared memory instead of global
  - [ ] Handle edge cases for tiles at image boundaries

- [ ] Benchmark and profile
  - [ ] Test on image sizes: 512×512, 1024×1024, 2048×2048
  - [ ] Measure kernel execution time (expect 2.8× speedup)
  - [ ] Profile with Nsight Compute: check global memory traffic reduction
  - [ ] Expected: ~670ms for 2048×2048 (down from ~2400ms)

##### Step 2: Separable Convolution (~300-350 LOC)
- [ ] Implement separable convolution in `backend/ops_conv2d.cu`
  - [ ] Create `convolution_row_kernel`: row-wise pass
    - [ ] Only horizontal apron needed
    - [ ] Shared memory: (TILE_W + 2×KERNEL_RADIUS) × TILE_H
    - [ ] Each thread loads left and right apron values
  - [ ] Create `convolution_col_kernel`: column-wise pass
    - [ ] Only vertical apron needed
    - [ ] Shared memory: TILE_W × (TILE_H + 2×KERNEL_RADIUS)
    - [ ] Each thread loads upper and lower apron values
  - [ ] Allocate intermediate buffer for row-pass results
  - [ ] Apply row kernel first, then column kernel

- [ ] Update `frontend/functional.py`
  - [ ] Add separable convolution path
  - [ ] Allocate and manage intermediate buffer
  - [ ] Call row kernel, then column kernel sequentially

- [ ] Implement backward pass for separable convolution
  - [ ] Reverse order: column backward, then row backward
  - [ ] Handle intermediate gradient buffer

- [ ] Benchmark and profile
  - [ ] Measure speedup (expect 6.2× over Step 1)
  - [ ] Expected: ~370ms for 2048×2048
  - [ ] Profile instruction count reduction
  - [ ] Verify correctness matches non-separable version

##### Step 3: Memory Access Optimization (~250-300 LOC)
- [ ] Reorganize shared memory layout in both kernels
  - [ ] Change from 2D array to 1D array layout
  - [ ] Update indexing: `data[y * width + x]` instead of `data[y][x]`
  - [ ] Ensure consecutive threads access consecutive memory addresses
  - [ ] Calculate proper strides for apron regions

- [ ] Eliminate shared memory bank conflicts
  - [ ] Ensure warp threads access different banks
  - [ ] Use proper memory alignment
  - [ ] May need padding in shared memory to avoid conflicts

- [ ] Add fast integer multiplication
  - [ ] Replace standard multiplication with `__mul24` for indices
  - [ ] Verify indices fit in 24-bit range

- [ ] Update both row and column kernels with new layout

- [ ] Benchmark and profile
  - [ ] Measure speedup (expect 3.2× over Step 2)
  - [ ] Expected: ~118ms for 2048×2048
  - [ ] **Total: 57× faster than naive implementation**
  - [ ] Profile with Nsight Compute: verify bank conflicts eliminated
  - [ ] Check memory bandwidth utilization improvement

##### Step 4: Advanced Optimizations (Optional) (~200-300 LOC)
- [ ] Texture memory optimization
  - [ ] Bind input tensor to texture memory
  - [ ] Use texture cache for input reads
  - [ ] Leverage automatic 2D caching

- [ ] Block size auto-tuning
  - [ ] Test different block configurations: 8×8, 16×16, 32×32
  - [ ] Test different shared memory tile sizes
  - [ ] Find optimal configuration for target GPU

- [ ] Loop unrolling
  - [ ] Add `#pragma unroll` directives to inner loops
  - [ ] Unroll convolution kernel loops
  - [ ] Measure impact on register usage

- [ ] Benchmark and profile
  - [ ] Measure additional speedup (expect 2× improvement)
  - [ ] Expected: ~58ms for 2048×2048
  - [ ] Compare against cuDNN (target: 80-90% of cuDNN performance)

##### Pooling Kernels (~200-250 LOC)
- [ ] Create `backend/ops_pooling.cu`
  - [ ] Implement `maxpool2d_forward_kernel`
    - [ ] Each thread computes one output position
    - [ ] Sliding window over input region
    - [ ] Find maximum value and its index
    - [ ] Store both output value and max indices
  - [ ] Implement `maxpool2d_backward_kernel`
    - [ ] Scatter gradients to max positions only
    - [ ] Use stored indices from forward pass
    - [ ] All other positions get zero gradient
  - [ ] Handle stride and padding correctly

- [ ] Optimize pooling kernels
  - [ ] Use shared memory for input tile caching
  - [ ] Reduce global memory accesses
  - [ ] Ensure coalesced memory access patterns

- [ ] Update `frontend/nn/pooling.py`
  - [ ] Dispatch to CUDA kernels when on GPU
  - [ ] Store max indices as tensor for backward pass

##### Testing & Validation (6B)
- [ ] Test each optimization step
  - [ ] Verify correctness against previous version
  - [ ] Ensure gradients still match PyTorch reference
  - [ ] Test on multiple image sizes

- [ ] Create comprehensive test suite
  - [ ] Test separable vs non-separable correctness
  - [ ] Test different padding modes
  - [ ] Test dilated convolutions (if implemented)
  - [ ] Gradcheck all optimized kernels

- [ ] Profile each step
  - [ ] Measure kernel execution time
  - [ ] Measure memory bandwidth utilization
  - [ ] Measure occupancy and register usage
  - [ ] Check for shared memory bank conflicts
  - [ ] Compare against cuDNN baseline

##### Example CNNs
- [ ] Update `examples/lenet.py`
  - [ ] Enable optimized kernels
  - [ ] Measure training time per epoch
  - [ ] Compare naive vs optimized performance
  - [ ] Report final accuracy on MNIST

- [ ] Create `examples/simple_cnn.py`
  - [ ] Implement modern CNN architecture
    - [ ] Conv→BatchNorm→ReLU→Pool pattern
    - [ ] Multiple convolutional blocks
    - [ ] Global average pooling
    - [ ] Dense classification head
  - [ ] Train on CIFAR-10 dataset
  - [ ] Benchmark training throughput (images/sec)
  - [ ] Compare CPU vs GPU speedup

##### Benchmarking & Profiling
- [ ] Create `benchmarks/conv_benchmark.py`
  - [ ] Benchmark each kernel variant separately
  - [ ] Test different image sizes: 32×32, 64×64, 128×128, 256×256
  - [ ] Test different kernel sizes: 3×3, 5×5, 7×7
  - [ ] Test different channel counts: 3, 16, 32, 64, 128
  - [ ] Plot performance graphs

- [ ] Compare against cuDNN
  - [ ] Benchmark equivalent PyTorch operations
  - [ ] Measure performance gap
  - [ ] Document where optimizations fall short

- [ ] Profile with Nsight Compute
  - [ ] Analyze memory bandwidth bottlenecks
  - [ ] Check arithmetic intensity
  - [ ] Identify optimization opportunities
  - [ ] Document hardware utilization metrics

**Stage 6B Milestones:**
- ✅ After Step 1: Practical CNN training with shared memory
- ✅ After Step 2-3: Fast CNN training, 57× faster than naive
- ✅ After Step 4: Near-optimal performance, within 80-90% of cuDNN

**Stage 6 Milestone:** ✅ LeNet achieves >99% accuracy on MNIST with optimized CUDA convolution kernels, comparable speed to cuDNN

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
- [ ] Document CUDA kernel implementations
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