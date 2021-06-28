#include "includes.h"
__global__ void elementwiseDiv(float *a, const float *b, const size_t len)
{
const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

if (idx >= len)
return;

a[idx] /= b[idx] + 1e-6f;
}