#include "includes.h"
__global__ void sqrt_kernel(float *v, int n) {
int x(threadIdx.x + blockDim.x * blockIdx.x);
if (x >= n) return;

v[x] = sqrt(v[x]);
}