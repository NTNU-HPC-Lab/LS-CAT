#include "includes.h"
__global__ void cuArraysCopyExtractVaryingOffsetCorr(const float *imageIn, const int inNX, const int inNY, float *imageOut, const int outNX, const int outNY, int *imageValid, const int nImages, const int2 *maxloc)
{

int idxImage = blockIdx.z;

int outx = threadIdx.x + blockDim.x*blockIdx.x;
int outy = threadIdx.y + blockDim.y*blockIdx.y;

int inx = outx + maxloc[idxImage].x - outNX/2;
int iny = outy + maxloc[idxImage].y - outNY/2;

if (outx < outNX && outy < outNY)
{
int idxOut = ( blockIdx.z * outNX + outx ) * outNY + outy;

int idxIn = ( blockIdx.z * inNX + inx ) * inNY + iny;

if (inx>=0 && iny>=0 && inx<inNX && iny<inNY) {

imageOut[idxOut] = imageIn[idxIn];
imageValid[idxOut] = 1;
}
else {
imageOut[idxOut] = 0.0f;
imageValid[idxOut] = 0;
}
}
}