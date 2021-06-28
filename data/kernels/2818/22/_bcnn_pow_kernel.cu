#include "includes.h"
__global__ void _bcnn_pow_kernel(int n, float *x, float a, float *y) {
int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
if (i < n) y[i] = pow(x[i], a);
}