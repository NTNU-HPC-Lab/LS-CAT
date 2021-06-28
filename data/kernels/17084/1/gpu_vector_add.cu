#include "includes.h"
__global__ void gpu_vector_add(float *out, float *a, float *b, int n) {
// built-in variable blockDim.x describes amount threads per block

int tid = blockIdx.x * blockDim.x + threadIdx.x;
if (tid < n)
out[tid] = a[tid] + b[tid];


// more advanced version - handling arbitrary vector/kernel size
// int i = blockIdx.x * blockDim.x + threadIdx.x;
// int step = gridDim.x * blockDim.x;

// for(; i < n; i += step){
//     out[i] = a[i] + b[i];
// }
}