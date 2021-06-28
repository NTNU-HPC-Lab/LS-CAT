#include "includes.h"
__global__ void updateGradientsKernel(float4 *D, float4 *TD, unsigned int nVertices)
{
int vidx = 4*(blockIdx.x * blockDim.x) + threadIdx.x;
int idx;
for (idx=0; idx<4*BLOCK_SIZE_AVGG; idx+=BLOCK_SIZE_AVGG)
{
D[vidx+idx] = TD[vidx+idx];
}
}