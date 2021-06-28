#include "includes.h"
__global__ void gpu_unmix32(int32_t * u, int32_t * v, int32_t * out, uint32_t stride, uint32_t * numSamples, int32_t * mixbits, int32_t * mixres, uint16_t * shiftUV, int32_t bytesShifted, int32_t theOutputPacketBytes, uint32_t frameLength)
{
int block = blockIdx.x % 8;
int index = blockIdx.x / 8;
int z = threadIdx.x + block * blockDim.x;
if (z < numSamples[index])
{
int32_t			shift = bytesShifted * 8;
int32_t		l, r;
int32_t k = z * 2;
int32_t * op = out + (index * theOutputPacketBytes) / 4;

if (mixres[index] != 0)
{
//Assert( bytesShifted != 0 );

/* matrixed stereo with shift */
int32_t		lt, rt;

lt = (u + index * frameLength)[z];
rt = (v + index * frameLength)[z];

l = lt + rt - ((mixres[index] * rt) >> mixbits[index]);
r = l - rt;

op += stride * z;
op[0] = (l << shift) | (uint32_t)(shiftUV + index * frameLength * 2)[k + 0];
op[1] = (r << shift) | (uint32_t)(shiftUV + index * frameLength * 2)[k + 1];
}
else
{
/* interleaving with shift */
op += stride * z;
op[0] = ((u + index * frameLength)[z] << shift) | (uint32_t)(shiftUV + index * frameLength * 2)[k + 0];
op[1] = ((v + index * frameLength)[z] << shift) | (uint32_t)(shiftUV + index * frameLength * 2)[k + 1];

}
}
}