#include "includes.h"
__global__ void gpu_mix16_1(int32_t * ip, uint32_t stride, int32_t * u, int32_t * v, int32_t numSamples, int32_t m2, int32_t mixbits, int32_t mixres)
{

int z = threadIdx.x + blockIdx.x * blockDim.x;

if (z < numSamples){

int32_t temp = ip[z];
int32_t		l, r;
l = (int16_t)temp;
r = temp >> 16;
u[z] = (mixres * l + m2 * r) >> mixbits;
v[z] = l - r;
}
}