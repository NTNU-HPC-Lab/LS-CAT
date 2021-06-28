#include "includes.h"
__global__ void add(float* vec_a, float* vec_b, float* vec_c, int n)
{
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < n)
{
vec_c[i] = vec_a[i] + vec_b[i];
i += blockDim.x * gridDim.x;
}
}