#include "includes.h"
__global__ void addScalarInArrayInPlace(float* in, float* add, float scale, int size) {
int tid = blockIdx.x * blockDim.x + threadIdx.x;
int stride = gridDim.x * blockDim.x;
for (; tid < size; tid += stride)
if (tid < size) in[tid] += add[0] * scale;
}