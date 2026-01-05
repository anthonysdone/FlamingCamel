// add
extern "C" __global__
void add_kernel(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x; 

    for (int i = idx; i < n; i += stride) {
        c[i] = a[i] + b[i]; 
    }
}

extern "C" __global__
void add_backward_kernel(float* grad_out, float* grad_a, float* grad_b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x; 

    for (int i = idx; i < n; i += stride) {
        grad_a[i] = grad_out[i];
        grad_b[i] = grad_out[i];
    }
}

// multiply
extern "C" __global__
void mul_kernel(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x; 

    for (int i = idx; i < n; i += stride) {
        c[i] = a[i] * b[i];
    }
}

extern "C" __global__
void mul_backward_kernel(float* grad_out, float* a, float* b, float* grad_a, float* grad_b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    int stride = gridDim.x * blockDim.x; 

    for (int i = idx; i < n; i += stride) {
        grad_a[i] = grad_out[i] * b[i]; 
        grad_b[i] = grad_out[i] * a[i]; 
    }
}

// relu
extern "C" __global__
void relu_kernel(float* x, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride) {
        out[i] = (x[i] > 0.0f) ? x[i] : 0.0f; 
    }
}

extern "C" __global__
void relu_backward_kernel(float* grad_out, float* x, float* grad_x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x; 

    for (int i = idx; i < n; i += stride) {
        grad_x[i] = (x[i] > 0.0f) ? grad_out[i] : 0.0f;
    }
}