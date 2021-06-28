#include "includes.h"
__global__ void windowBartlett(float* idata, int length)
{
int tidx = threadIdx.x + blockIdx.x*blockDim.x;
if (tidx < length)
{
idata[tidx] = 0;
}
}