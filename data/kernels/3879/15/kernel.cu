#include "includes.h"
__global__ void kernel(const uint *__restrict__ a, const uint *__restrict__ b, const uint length, uint *c)
{
uint tid = (blockIdx.x * blockDim.x) + threadIdx.x;
const uint stride = blockDim.x * gridDim.x;
while (tid < length)
{
c[tid] = a[tid] + b[tid];
tid += stride;
}
}