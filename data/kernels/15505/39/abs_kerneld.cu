#include "includes.h"
__global__ void abs_kerneld(double *v, int n) {
int x(threadIdx.x + blockDim.x * blockIdx.x);
if (x >= n) return;

v[x] = ::abs(v[x]);
}