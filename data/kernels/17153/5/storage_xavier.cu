#include "includes.h"
__global__ void storage_xavier(float *a, int size, float scale, curandState *cs) {
int index = blockIdx.x * blockDim.x + threadIdx.x;
if (index < size) {
curand_init(1234, index, 0, &cs[index]);
a[index] = (curand_uniform(&cs[index]) * 2 - 1) * scale;
}
}