#include "includes.h"
__global__ void kernelUpdateParticle(double *positions, double *velocities, double *pBests, double *gBest, int particlesCount, int dimensionsCount, double r1, double r2)
{
int i = blockIdx.x * blockDim.x + threadIdx.x;

if(i >= particlesCount * dimensionsCount)
return;

velocities[i] = d_OMEGA * velocities[i] + r1 * (pBests[i] - positions[i])
+ r2 * (gBest[i % dimensionsCount] - positions[i]);

// Update posisi particle
positions[i] += velocities[i];
}