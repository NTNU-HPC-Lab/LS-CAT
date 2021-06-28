#include "includes.h"
__global__ void gpu_mix24_2_1(uint8_t * ip, uint32_t stride, int32_t * u, int32_t * v, int32_t numSamples, uint16_t * shiftUV, uint32_t mask, int32_t shift)
{
int z = threadIdx.x + blockIdx.x * blockDim.x;
if (z < numSamples)
{
int32_t		l, r;
int32_t k = z * 2;


ip += 3 * z;
ip += (stride - 1) * 3 * z;
l = (int32_t)(((uint32_t)ip[HBYTE] << 16) | ((uint32_t)ip[MBYTE] << 8) | (uint32_t)ip[LBYTE]);
l = (l << 8) >> 8;

ip += 3;
r = (int32_t)(((uint32_t)ip[HBYTE] << 16) | ((uint32_t)ip[MBYTE] << 8) | (uint32_t)ip[LBYTE]);
r = (r << 8) >> 8;

shiftUV[k + 0] = (uint16_t)(l & mask);
shiftUV[k + 1] = (uint16_t)(r & mask);

l >>= shift;
r >>= shift;

u[z] = l;
v[z] = r;
}
}