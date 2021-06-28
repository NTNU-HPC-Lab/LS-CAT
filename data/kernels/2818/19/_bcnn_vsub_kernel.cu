#include "includes.h"
__global__ void _bcnn_vsub_kernel(int n, float *a, float *b, float *y) {
int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
if (i < n) y[i] = a[i] - b[i];
}