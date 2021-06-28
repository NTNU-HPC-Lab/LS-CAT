#include "includes.h"
__global__ void devFindUniqueTriangleAffected(int maxIndex, int *pTriangleAffected, int *pTriangleAffectedIndex, int *pUniqueFlag)
{
unsigned int i = blockIdx.x*blockDim.x + threadIdx.x + 1;

while (i < maxIndex) {
if (pTriangleAffected[i-1] == pTriangleAffected[i] &&
pTriangleAffected[i] != -1) {
int j = pTriangleAffectedIndex[i];
pUniqueFlag[j] = 0;
}

i += gridDim.x*blockDim.x;
}
}