#include "includes.h"
__global__ void devInverseReindexInt(int N, int *destArray, int *srcArray, int *reindex, int realSize, int nDims)
{
for (unsigned int n = 0; n < nDims; n++) {
int i = blockIdx.x*blockDim.x + threadIdx.x;

while (i < N) {
int tmp = srcArray[i + n*realSize];
destArray[i + n*realSize] = reindex[tmp];

i += gridDim.x*blockDim.x;
}
}
}