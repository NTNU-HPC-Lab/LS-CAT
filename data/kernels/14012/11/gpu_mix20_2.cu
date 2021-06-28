#include "includes.h"
__global__ void gpu_mix20_2(uint8_t * ip, uint32_t stride, int32_t * u, int32_t * v, int32_t numSamples)
{
int z = threadIdx.x + blockIdx.x * blockDim.x;
if (z < numSamples)
{
int32_t		l, r;


ip += 3 * z;
ip += (stride - 1) * 3 * z;
l = (int32_t)(((uint32_t)ip[HBYTE] << 16) | ((uint32_t)ip[MBYTE] << 8) | (uint32_t)ip[LBYTE]);
u[z] = (l << 8) >> 12;

ip += 3;
r = (int32_t)(((uint32_t)ip[HBYTE] << 16) | ((uint32_t)ip[MBYTE] << 8) | (uint32_t)ip[LBYTE]);
v[z] = (r << 8) >> 12;
}
}