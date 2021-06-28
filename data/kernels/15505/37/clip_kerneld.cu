#include "includes.h"
__global__ void clip_kerneld(double *v, int n, double limit) {
int x(threadIdx.x + blockDim.x * blockIdx.x);
if (x >= n) return;

v[x] = (v[x] > limit) ? limit : ((v[x] < -limit) ? -limit : v[x]);
}