#include "includes.h"
__global__ void normalizeGradient(float* gradient, int* activeMask, int activePatches, int patches)
{
int i = threadIdx.x + blockIdx.x * blockDim.x;
if (i >= activePatches)
return;

int patch = activeMask[i];

float norm = gradient[6 * patches + patch];
if (norm > 0)
norm = 1.0f / sqrtf(norm);

for (int j = 0; j < 6; ++j)
gradient[j*patches + patch] *= norm;
}