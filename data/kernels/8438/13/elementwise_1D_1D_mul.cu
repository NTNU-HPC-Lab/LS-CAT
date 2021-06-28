#include "includes.h"

using namespace std;
#ifndef MAP_FILE
#define MAP_FILE MAP_SHARED
#endif

__global__ void elementwise_1D_1D_mul(float* in1, float* in2, float* out, int size) {
int tid = blockIdx.x * blockDim.x + threadIdx.x;
int stride = gridDim.x * blockDim.x;
for (; tid < size; tid += stride)
if (tid < size) out[tid] = in1[tid] * in2[tid];
}