#include "includes.h"
__global__ void pow_kernel(float *v, int n, float e) {
int x(threadIdx.x + blockDim.x * blockIdx.x);
if (x >= n) return;

v[x] = ::pow(v[x], e);
}