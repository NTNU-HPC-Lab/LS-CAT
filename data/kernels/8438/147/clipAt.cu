#include "includes.h"
__global__ void clipAt(float* in, float bound, int size) {
int tid = blockIdx.x * blockDim.x + threadIdx.x;
int stride = gridDim.x * blockDim.x;
for (; tid < size; tid += stride)
if (tid < size) {
if (in[tid] > bound) in[tid] = bound;
if (in[tid] < -bound) in[tid] = -bound;
}
}