#include "includes.h"
__global__ void kernelMarkDeadTriangles(int *cmarker, short *cnewtri, int nTris) {
int x = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

if (x >= nTris)
return ;

cmarker[x] = (cnewtri[x] >= 0 ? 0 : 1);
}