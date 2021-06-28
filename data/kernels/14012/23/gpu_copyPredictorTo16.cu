#include "includes.h"
__global__ void gpu_copyPredictorTo16(int32_t * in, int16_t * out, uint32_t stride, uint32_t * numSamples, int32_t theOutputPacketBytes, uint32_t frameLength)
{
int block = blockIdx.x % 8;
int index = blockIdx.x / 8;
int z = threadIdx.x + block * blockDim.x;

if (z < numSamples[index])
{
int16_t * op = out + (index * theOutputPacketBytes) / 2;

op[z] = (int16_t)(in + index * frameLength)[z];
}
}