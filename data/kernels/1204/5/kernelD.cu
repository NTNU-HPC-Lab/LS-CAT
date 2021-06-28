#include "includes.h"
__global__ void kernelD(int n, float *x, float *y) {
int index = blockIdx.x * blockDim.x + threadIdx.x;
int stride = blockDim.x * gridDim.x;
for (int i = index; i < n; i += stride) {
for (int j = 0; j < n / CONST; j++)
y[i] = atomicAdd(&y[j], x[j]);
}
}