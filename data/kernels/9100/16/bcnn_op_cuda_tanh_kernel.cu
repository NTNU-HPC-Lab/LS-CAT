#include "includes.h"
__global__ void bcnn_op_cuda_tanh_kernel(int n, float *x, float *y)
{
int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
if (i < n) {
y[i] = (exp(2 * x[i]) - 1) / (exp(2 * x[i]) + 1);
}
return;
}