#include "includes.h"
__global__ void cuArraysCopyToBatchWithOffset_kernel(const float2 *imageIn, const int inNY, float2 *imageOut, const int outNX, const int outNY, const int nImages, const int *offsetX, const int *offsetY)
{
int idxImage = blockIdx.z;
int outx = threadIdx.x + blockDim.x*blockIdx.x;
int outy = threadIdx.y + blockDim.y*blockIdx.y;
if(idxImage>=nImages || outx >= outNX || outy >= outNY) return;
int idxOut = idxImage*outNX*outNY + outx*outNY + outy;
int idxIn = (offsetX[idxImage]+outx)*inNY + offsetY[idxImage] + outy;
imageOut[idxOut] = imageIn[idxIn];
}