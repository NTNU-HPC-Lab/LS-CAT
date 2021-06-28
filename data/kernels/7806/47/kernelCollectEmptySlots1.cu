#include "includes.h"
__global__ void kernelCollectEmptySlots1(short *cnewtri, int *cprefix, int *cempty, int nTris)
{
int x = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

if (x >= nTris || cnewtri[x] >= 0)
return ;

int id = x - cprefix[x];

cempty[id] = x;
}