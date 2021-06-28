#include "includes.h"
__global__ void gpu_mix24_1_2(uint8_t * ip, uint32_t stride, int32_t * u, int32_t * v, int32_t numSamples, int32_t mixres, int32_t m2, int32_t mixbits)
{
int z = threadIdx.x + blockIdx.x * blockDim.x;
if (z < numSamples)
{
int32_t		l, r;

ip += 3 * z;
ip += (stride - 1) * 3 * z;
l = (int32_t)(((uint32_t)ip[HBYTE] << 16) | ((uint32_t)ip[MBYTE] << 8) | (uint32_t)ip[LBYTE]);
l = (l << 8) >> 8;

ip += 3;
r = (int32_t)(((uint32_t)ip[HBYTE] << 16) | ((uint32_t)ip[MBYTE] << 8) | (uint32_t)ip[LBYTE]);
r = (r << 8) >> 8;

u[z] = (mixres * l + m2 * r) >> mixbits;
v[z] = l - r;
}
}