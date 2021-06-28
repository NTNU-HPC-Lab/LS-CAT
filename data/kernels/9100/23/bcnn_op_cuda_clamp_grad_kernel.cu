#include "includes.h"
__global__ void bcnn_op_cuda_clamp_grad_kernel(int n, float *x, float *dx)
{
int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
if (i < n) {
dx[i] *= (float)(x[i] > 0.0f && (x[i] < 1.0f));
}
return;
}