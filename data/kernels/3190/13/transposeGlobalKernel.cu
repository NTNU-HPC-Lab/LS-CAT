#include "includes.h"
__global__ void transposeGlobalKernel(float* idata, float* odata, int width, int height)
{
int tidx = blockIdx.x * blockDim.x + threadIdx.x;
int tidy = blockIdx.y * blockDim.y+ threadIdx.y;

if(tidx < width && tidy < height)
{
odata[tidx*height + tidy] = idata[tidy*width + tidx];
}
}