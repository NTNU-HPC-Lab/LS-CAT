#include "includes.h"
__global__ void pow_kerneld(double *v, int n, double e) {
int x(threadIdx.x + blockDim.x * blockIdx.x);
if (x >= n) return;

v[x] = ::pow(v[x], e);
}