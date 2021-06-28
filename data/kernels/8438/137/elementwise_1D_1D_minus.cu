#include "includes.h"
__global__ void elementwise_1D_1D_minus(float* in1, float* in2, float* out, int size) {
int tid = blockIdx.x * blockDim.x + threadIdx.x;
int stride = gridDim.x * blockDim.x;
for (; tid < size; tid += stride)
if (tid < size) out[tid] = in1[tid] - in2[tid];
}