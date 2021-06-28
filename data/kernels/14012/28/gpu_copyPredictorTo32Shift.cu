#include "includes.h"
__global__ void gpu_copyPredictorTo32Shift(int32_t * in, uint16_t * shift, int32_t * out, uint32_t stride, uint32_t * numSamples, int32_t bytesShifted, int32_t theOutputPacketBytes, uint32_t frameLength)
{

int block = blockIdx.x % 8;
int index = blockIdx.x / 8;
int z = threadIdx.x + block * blockDim.x;

if (z < numSamples[index])
{
int32_t * op = out + (index * theOutputPacketBytes) / 4;
int32_t	shiftVal = bytesShifted * 8;

//Assert( bytesShifted != 0 );

op += stride * z;
op[0] = ((in + index * frameLength)[z] << shiftVal) | (uint32_t)(shift + index * frameLength * 2)[z];

}
}