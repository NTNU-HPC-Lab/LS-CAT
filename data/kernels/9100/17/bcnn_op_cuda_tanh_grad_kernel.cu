#include "includes.h"
__global__ void bcnn_op_cuda_tanh_grad_kernel(int n, float *x, float *dx)
{
int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
if (i < n) {
dx[i] *= (1 - x[i] * x[i]);
}
return;
}