set -e

echo "Compiling CUDA kernels..."

if [ -f backend/ops_elementwise.cu ]; then
    echo "Compiling ops_elementwise.cu..."
    nvcc -ptx backend/ops_elementwise.cu -o backend/ops_elementwise.ptx
    echo "Ops_elementwise.cu compiled to ops_elementwise.ptx!"
fi

echo "All kernels compiled successfully!" 
