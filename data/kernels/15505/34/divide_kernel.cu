#include "includes.h"
__global__ void divide_kernel(float *v, float *other, int n) {
int x(threadIdx.x + blockDim.x * blockIdx.x);
if (x >= n) return;

v[x] /= other[x];
}