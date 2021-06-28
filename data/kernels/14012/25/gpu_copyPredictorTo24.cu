#include "includes.h"
__global__ void gpu_copyPredictorTo24(int32_t * in, uint8_t * out, uint32_t stride, uint32_t * numSamples, int32_t theOutputPacketBytes, uint32_t frameLength)
{
int block = blockIdx.x % 8;
int index = blockIdx.x / 8;
int z = threadIdx.x + block * blockDim.x;

if (z < numSamples[index])
{
uint8_t * op = out + (index * theOutputPacketBytes);
int32_t	val = (in + index * frameLength)[z];
op += (stride * 3 * z);

op[HBYTE] = (uint8_t)((val >> 16) & 0xffu);
op[MBYTE] = (uint8_t)((val >> 8) & 0xffu);
op[LBYTE] = (uint8_t)((val >> 0) & 0xffu);
}
}