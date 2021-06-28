#include "includes.h"
__global__ void windowKernel(float* idata, float* window, int width, int height)
{
int tidx = threadIdx.x + blockIdx.x*blockDim.x;
int tidy = threadIdx.y + blockIdx.y*blockDim.y;
if(tidx < width && tidy < height)
{
idata[tidy * width + tidx] = window[tidx] * idata[tidy * width + tidx];
}
}