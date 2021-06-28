#include "includes.h"
extern "C"

// don't forget to compile with "nvcc -ptx cudaKernel.cu -o cudaKernel.ptx
// And to move the ptx file in the resources !
__global__ void add(int n, float* a, float* b, float* sum) {
int index = blockIdx.x * blockDim.x + threadIdx.x;
int stride = blockDim.x * gridDim.x;
for (int i = index; i < n; i += stride)
sum[i] = a[i] + b[i];
}