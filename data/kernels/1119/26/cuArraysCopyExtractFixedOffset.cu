#include "includes.h"
__global__ void cuArraysCopyExtractFixedOffset(const float *imageIn, const int inNX, const int inNY, float *imageOut, const int outNX, const int outNY, const int nImages, const int offsetX, const int offsetY)
{
int outx = threadIdx.x + blockDim.x*blockIdx.x;
int outy = threadIdx.y + blockDim.y*blockIdx.y;

if(outx < outNX && outy < outNY)
{
int idxOut = (blockIdx.z * outNX + outx)*outNY+outy;
int idxIn = (blockIdx.z*inNX + outx + offsetX)*inNY + outy + offsetY;
imageOut[idxOut] = imageIn[idxIn];
}
}