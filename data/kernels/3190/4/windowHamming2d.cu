#include "includes.h"
__global__ void windowHamming2d(float* idata, int length, int height)
{
int tidx = threadIdx.x + blockIdx.x*blockDim.x;
int tidy = threadIdx.y + blockIdx.y*blockDim.y;
//printf("tidy: %d, tidy:%d, idx:%d", tidy,tidx ,tidy * length + tidx);
if (tidx < length && tidy < height)
{
//printf("tidy: %d, tidy:%d, idx:%d", tidy,tidx ,tidy * length + tidx);
idata[tidy * length + tidx] = (0.54 - 0.46 * cos(2*tidy*PI_F / (height - 1))) * (0.54 - 0.46 * cos(2*tidx*PI_F / (length - 1)));
}
}