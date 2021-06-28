#include "includes.h"
__global__ void kernelFixVertArray(int *ctriangles, int nTris, int *cvertarr)

{
int x = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
if (x >= nTris)
return ;

int v0 = ctriangles[x * 9 + 4];
int v1 = ctriangles[x * 9 + 5];
int v2 = ctriangles[x * 9 + 3];

ctriangles[x * 9 + 6] = atomicExch(&cvertarr[v0], (x << 2));
ctriangles[x * 9 + 7] = atomicExch(&cvertarr[v1], (x << 2) | 1);
ctriangles[x * 9 + 8] = atomicExch(&cvertarr[v2], (x << 2) | 2);
}