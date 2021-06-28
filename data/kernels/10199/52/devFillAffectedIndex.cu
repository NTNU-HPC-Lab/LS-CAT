#include "includes.h"
__global__ void devFillAffectedIndex(int nRemove, int maxTriPerVert, int *pTriangleAffectedIndex)
{
int n = blockIdx.x*blockDim.x + threadIdx.x;

while (n < nRemove) {
for (int i = 0; i < maxTriPerVert; i++) {
pTriangleAffectedIndex[i + n*maxTriPerVert] = n;
pTriangleAffectedIndex[i + n*maxTriPerVert + nRemove*maxTriPerVert] = n;
}

n += blockDim.x*gridDim.x;
}
}