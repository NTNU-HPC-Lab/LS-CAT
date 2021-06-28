#include "includes.h"
__global__ void vector_add(float *out, float *a, float *b, int n) {
int tid = blockIdx.x * blockDim.x + threadIdx.x;

// Handling arbitrary vector size
if (tid < n){
out[tid] = a[tid] + b[tid];
}
}