#include "includes.h"
__global__ void computeGradientCentralDiff(const float* similarities, float* gradient, int* activeMask, int activePatches, int patches, int p)
{
int i = threadIdx.x + blockIdx.x * blockDim.x;
if (i >= activePatches)
return;
int patch = activeMask[i];

float dx = similarities[patch] - similarities[patches + patch];
gradient[p*patches + patch] = dx;
if (p == 0)
gradient[6 * patches + patch] = dx*dx;
else
gradient[6 * patches + patch] += dx*dx;
}