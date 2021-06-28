#include "includes.h"
__global__ void gpu_mix32_2_2(int64_t * ip, uint32_t stride, int32_t * u, int32_t * v, int32_t numSamples, uint16_t * shiftUV, uint32_t mask, int32_t shift)
{
int z = threadIdx.x + blockIdx.x * blockDim.x;
if (z < numSamples)
{
int32_t		l, r;
int32_t k = z * 2;


int64_t temp = ip[z];


l = (int32_t)temp;
r = temp >> 32;

shiftUV[k + 0] = (uint16_t)(l & mask);
shiftUV[k + 1] = (uint16_t)(r & mask);

l >>= shift;
r >>= shift;

u[z] = l;
v[z] = r;
}
}