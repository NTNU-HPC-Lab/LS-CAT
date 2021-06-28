#include "includes.h"
__global__ void devFillAffectedTriangles(int nFlip, int *pTaff, int *pTaffEdge, int *pEnd, int2 *pEt)
{
unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

while (i < nFlip) {
int e = pEnd[i];

pTaffEdge[i] = i;
pTaffEdge[i + nFlip] = i;

pTaff[i]         = pEt[e].x;
pTaff[i + nFlip] = pEt[e].y;

i += gridDim.x*blockDim.x;
}
}