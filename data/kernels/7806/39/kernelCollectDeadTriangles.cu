#include "includes.h"
__global__ void kernelCollectDeadTriangles(int *cdeadTri, short *cnewtri, int *cmarker, int nTris) {
int x = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

if (x >= nTris || cnewtri[x] >= 0)
return ;

int id = cmarker[x];

cdeadTri[id] = x;
}