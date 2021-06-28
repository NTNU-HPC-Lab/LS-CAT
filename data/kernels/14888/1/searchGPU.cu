#include "includes.h"
__device__ float f(float x)
{
return 4.f / (1.f + x * x);
}
__global__ void searchGPU(float *data, const float x, int *result)
{
int idx = blockIdx.x * blockDim.x + threadIdx.x;

if (data[idx] == x)
{
result[0] = static_cast<int>(data[idx]);
result[1] = idx;
}
}