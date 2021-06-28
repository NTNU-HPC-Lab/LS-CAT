#include "includes.h"

using namespace std;
#ifndef MAP_FILE
#define MAP_FILE MAP_SHARED
#endif

__global__ void adagrad_update_1D_1D(float* x, float* d, float* m, float clip, float lr, int size) {
int tid = blockIdx.x * blockDim.x + threadIdx.x;
int stride = gridDim.x * blockDim.x;
for (; tid < size; tid += stride) {
if (d[tid] > clip) d[tid] = clip;
if (d[tid] < -clip) d[tid] = -clip;
m[tid] += d[tid] * d[tid];
x[tid] -= lr * d[tid] / sqrt(m[tid] + 0.00000001);
d[tid] = 0;
}
}