#include "includes.h"
__global__ void gpu_unmix16(int32_t * u, int32_t * v, int16_t * out, uint32_t stride, uint32_t * numSamples, int32_t * mixbits, int32_t * mixres, int32_t theOutputPacketBytes, uint32_t frameLength)
{
int block = blockIdx.x % 8;
int index = blockIdx.x / 8;
int z = threadIdx.x + block * blockDim.x;

if (z < numSamples[index])
{

int32_t		l, r;
int16_t * op = out + (index * theOutputPacketBytes) / 2;

if (mixres[index] != 0)
{
/* matrixed stereo */
l = (u + index * frameLength)[z] + (v + index * frameLength)[z] - ((mixres[index] * (v + index * frameLength)[z]) >> mixbits[index]);
r = l - (v + index * frameLength)[z];
op += stride* z;
op[0] = (int16_t)l;
op[1] = (int16_t)r;
}
else
{
/* Conventional separated stereo. */
op += stride * z;
op[0] = (int16_t)(u + index * frameLength)[z];
op[1] = (int16_t)(v + index * frameLength)[z];
}

}
}