#include "includes.h"
__global__ void windowBartlett2d(float* idata, int length, int height)
{
int tidx = threadIdx.x + blockIdx.x*blockDim.x;
int tidy = threadIdx.y + blockIdx.y*blockDim.y;
if (tidx < length && tidy < height)
{
idata[tidy * length + tidx] = 0;
}
}