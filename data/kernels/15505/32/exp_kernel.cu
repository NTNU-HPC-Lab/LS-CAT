#include "includes.h"
__global__ void exp_kernel(float *v, int n) {
int x(threadIdx.x + blockDim.x * blockIdx.x);
if (x >= n) return;

v[x] = exp(v[x]);
}