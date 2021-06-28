#include "includes.h"
__global__ void kernelMarkValidTriangles(short *cnewtri, int *cvalid, int nTris)
{
int x = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

if (x >= nTris)
return ;

cvalid[x] = (cnewtri[x] >= 0) ? 1 : 0;
}