#include "includes.h"
__global__ void gpu_touch(uint64_t *x, const size_t size)
{
const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < size)
{
x[i] = 0;
}
}