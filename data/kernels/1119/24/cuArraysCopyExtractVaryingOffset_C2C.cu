#include "includes.h"
__global__ void cuArraysCopyExtractVaryingOffset_C2C(const float2 *imageIn, const int inNX, const int inNY, float2 *imageOut, const int outNX, const int outNY, const int nImages, const int2 *offsets)
{
int outx = threadIdx.x + blockDim.x*blockIdx.x;
int outy = threadIdx.y + blockDim.y*blockIdx.y;

if(outx < outNX && outy < outNY)
{
int idxImage = blockIdx.z;
int idxOut = (blockIdx.z * outNX + outx)*outNY+outy;
int idxIn = (blockIdx.z*inNX + outx + offsets[idxImage].x)*inNY + outy + offsets[idxImage].y;
imageOut[idxOut] = imageIn[idxIn];
}
}