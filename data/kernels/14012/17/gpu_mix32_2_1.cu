#include "includes.h"
__global__ void gpu_mix32_2_1(int64_t * ip, uint32_t stride, int32_t * u, int32_t * v, int32_t numSamples)
{
int z = threadIdx.x + blockIdx.x * blockDim.x;
if (z < numSamples)
{
int64_t temp = ip[z];

u[z] = (int32_t)temp;
v[z] = temp >> 32;
}
}