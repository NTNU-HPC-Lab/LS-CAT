#include "includes.h"

__global__ void kernel(float* indata, float* outdata)
{
const auto i = blockIdx.x * blockDim.x + threadIdx.x;
const auto j = blockIdx.y * blockDim.y + threadIdx.y;

if (i >= Size[0] || j >= Size[1])
return;

outdata[j + i * Size[1]] = indata[j + i * Size[1]] * (Size[0] + Spacing[1]);
printf("[%u,%u] -> %.2f -> %.2f\n", i, j, indata[j + i * Size[1]], outdata[j + i * Size[1]]);
}