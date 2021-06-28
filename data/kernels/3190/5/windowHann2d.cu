#include "includes.h"
__global__ void windowHann2d(float* idata, int length, int height)
{
int tidx = threadIdx.x + blockIdx.x*blockDim.x;
int tidy = threadIdx.y + blockIdx.y*blockDim.y;
if (tidx < length && tidy < height)
{
idata[tidy * length + tidx] =  0.5*(1 + cos(2*tidy*PI_F / (height - 1))) * 0.5*(1 + cos(2*tidx*PI_F / (length - 1)));
}
}