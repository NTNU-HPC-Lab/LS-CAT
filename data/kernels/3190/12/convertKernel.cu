#include "includes.h"
__global__ void convertKernel(short* idata, float* odata, int size)
{
int tidx = threadIdx.x + blockIdx.x*blockDim.x;
if(tidx < size)
odata[tidx] = (float)idata[tidx];
}