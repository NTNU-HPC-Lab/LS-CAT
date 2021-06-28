#include "includes.h"
__global__ void iterative_saxpy_kernel(float *y, const float* x, const float alpha, const float beta, int n_loop)
{
int idx = blockIdx.x * blockDim.x + threadIdx.x;

for (int i = 0; i < n_loop; i++)
y[idx] = alpha * x[idx] + beta;
}