#include "includes.h"
__global__ void divScalar(float* in, float* out, float div, int size) {
int tid = blockIdx.x * blockDim.x + threadIdx.x;
int stride = gridDim.x * blockDim.x;
for (; tid < size; tid += stride)
if (tid < size) out[tid] = in[tid] / div;
}