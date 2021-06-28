#include "includes.h"
__global__ void gpu_unmix20(int32_t * u, int32_t * v, uint8_t * out, uint32_t stride, uint32_t * numSamples, int32_t * mixbits, int32_t * mixres, int32_t theOutputPacketBytes, uint32_t frameLength)
{
int block = blockIdx.x % 8;
int index = blockIdx.x / 8;
int z = threadIdx.x + block * blockDim.x;

if (z < numSamples[index])
{

int32_t		l, r;
uint8_t * op = out + (index * theOutputPacketBytes) ;

if (mixres[index] != 0)
{
/* matrixed stereo */
l = (u + index * frameLength)[z] + (v + index * frameLength)[z] - ((mixres[index] * (v + index * frameLength)[z]) >> mixbits[index]);
r = l - (v + index * frameLength)[z];

l <<= 4;
r <<= 4;

op += 3 * z;
op += (stride - 1) * 3 * z;
op[HBYTE] = (uint8_t)((l >> 16) & 0xffu);
op[MBYTE] = (uint8_t)((l >> 8) & 0xffu);
op[LBYTE] = (uint8_t)((l >> 0) & 0xffu);
op += 3;

op[HBYTE] = (uint8_t)((r >> 16) & 0xffu);
op[MBYTE] = (uint8_t)((r >> 8) & 0xffu);
op[LBYTE] = (uint8_t)((r >> 0) & 0xffu);
}
else
{
/* Conventional separated stereo. */
int32_t		val;

val = (u + index * frameLength)[z] << 4;
op += 3 * z;
op += (stride - 1) * 3 * z;
op[HBYTE] = (uint8_t)((val >> 16) & 0xffu);
op[MBYTE] = (uint8_t)((val >> 8) & 0xffu);
op[LBYTE] = (uint8_t)((val >> 0) & 0xffu);
op += 3;

val = (v + index * frameLength)[z] << 4;
op[HBYTE] = (uint8_t)((val >> 16) & 0xffu);
op[MBYTE] = (uint8_t)((val >> 8) & 0xffu);
op[LBYTE] = (uint8_t)((val >> 0) & 0xffu);
}
}
}