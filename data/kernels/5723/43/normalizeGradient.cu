#include "includes.h"
__global__ void normalizeGradient(float* gradient, int* activeMask, int activeSlices, int slices)
{
int i = threadIdx.x + blockIdx.x * blockDim.x;
if (i >= activeSlices)
return;

int slice = activeMask[i];

float norm = gradient[6 * slices + slice];
if (norm > 0)
norm = 1.0f / sqrtf(norm);

for (int j = 0; j < 6; ++j)
gradient[j*slices + slice] *= norm;
}