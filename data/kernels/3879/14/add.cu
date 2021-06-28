#include "includes.h"
__global__ void add(const float3 *__restrict__ dFinalForce, const unsigned int noRainDrops, float3 *__restrict__ dRainDrops)
{
//TODO: Add the FinalForce to every Rain drops position.
uint xOffset = (blockIdx.x * blockDim.x) + threadIdx.x;
uint xSkip = gridDim.x * blockDim.x;

while (xOffset < noRainDrops)
{
dRainDrops[xOffset].x += dFinalForce->x;
dRainDrops[xOffset].y += dFinalForce->y;
dRainDrops[xOffset].z += dFinalForce->z;

xOffset += xSkip;
}
}