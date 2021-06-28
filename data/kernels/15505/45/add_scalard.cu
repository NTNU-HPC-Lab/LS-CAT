#include "includes.h"
__global__ void add_scalard(double *v, double other, int n) {
int x(threadIdx.x + blockDim.x * blockIdx.x);
if (x >= n) return;

v[x] += other;
}