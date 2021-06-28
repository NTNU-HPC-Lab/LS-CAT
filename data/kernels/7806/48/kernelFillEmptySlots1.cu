#include "includes.h"
__global__ void kernelFillEmptySlots1(short *cnewtri, int *cprefix, int *cempty, int *ctriangles, int nTris, int newnTris, int offset)
{
int x = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

if (x >= nTris || cnewtri[x] < 0)
return ;

int value;

if (x < newnTris)
value = x;
else {
value = cempty[cprefix[x] - offset];

for (int i = 0; i < 9; i++)
ctriangles[value * 9 + i] = ctriangles[x * 9 + i];
}

cprefix[x] = value;
}