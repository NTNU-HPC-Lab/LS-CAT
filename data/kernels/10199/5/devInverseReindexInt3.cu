#include "includes.h"
__global__ void devInverseReindexInt3(int N, int3 *destArray, int3 *srcArray, int *reindex, int realSize, int nDims)
{
for (unsigned int n = 0; n < nDims; n++) {
int i = blockIdx.x*blockDim.x + threadIdx.x;

while (i < N) {
int tmp = srcArray[i + n*realSize].x;
destArray[i + n*realSize].x = reindex[tmp];
tmp = srcArray[i + n*realSize].y;
destArray[i + n*realSize].y = reindex[tmp];
tmp = srcArray[i + n*realSize].z;
destArray[i + n*realSize].z = reindex[tmp];

i += gridDim.x*blockDim.x;
}
}
}