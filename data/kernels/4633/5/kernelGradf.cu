#include "includes.h"
__global__ void kernelGradf(const float *d_x, float *d_grad, float *A, float *b, const size_t len)
{
size_t index = blockIdx.x * blockDim.x + threadIdx.x;

if (index >= len)
return;

d_grad[index] = 0.0f;

for (size_t j = 0; j < len; ++j)
{
d_grad[index] += A[index * len + j] * d_x[j];
}

d_grad[index] *= 2.0f;
d_grad[index] += b[index];
}