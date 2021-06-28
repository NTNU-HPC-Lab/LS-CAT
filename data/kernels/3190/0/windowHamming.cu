#include "includes.h"
__global__ void windowHamming(float* idata, int length)
{
int tidx = threadIdx.x + blockIdx.x*blockDim.x;
if (tidx < length)
{
idata[tidx] = 0.54 - 0.46 * cos(2*tidx*PI_F / (length - 1));
}
}