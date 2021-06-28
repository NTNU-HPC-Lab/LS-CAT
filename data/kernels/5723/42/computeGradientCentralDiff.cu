#include "includes.h"
__global__ void computeGradientCentralDiff(const float* similarities, float* gradient, int* activeMask, int activeSlices, int slices, int p)
{
int i = threadIdx.x + blockIdx.x * blockDim.x;
if (i >= activeSlices)
return;
int slice = activeMask[i];

float dx = similarities[slice] - similarities[slices + slice];
gradient[p*slices + slice] = dx;
if (p == 0)
gradient[6 * slices + slice] = dx*dx;
else
gradient[6 * slices + slice] += dx*dx;
}