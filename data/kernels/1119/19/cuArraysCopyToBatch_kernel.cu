#include "includes.h"
__global__ void cuArraysCopyToBatch_kernel(const float2 *imageIn, const int inNX, const int inNY, float2 *imageOut, const int outNX, const int outNY, const int nImagesX, const int nImagesY, const int strideX, const int strideY)
{
int idxImage = blockIdx.z;
int outx = threadIdx.x + blockDim.x*blockIdx.x;
int outy = threadIdx.y + blockDim.y*blockIdx.y;
if(idxImage >=nImagesX*nImagesY|| outx >= outNX || outy >= outNY) return;
int idxOut = idxImage*outNX*outNY + outx*outNY + outy;
int idxImageX = idxImage/nImagesY;
int idxImageY = idxImage%nImagesY;
int idxIn = (idxImageX*strideX+outx)*inNY + idxImageY*strideY+outy;
imageOut[idxOut] = imageIn[idxIn];
}