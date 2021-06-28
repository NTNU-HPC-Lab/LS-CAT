#include "includes.h"
__global__ void Compute_weightx_weighty2_norm0_Kernel(float* weightx, float* weighty, const float* absIx, const float* absIy, int nPixels, float norm_for_smooth_term, float eps)
{
int bx = blockIdx.x;
int tx = threadIdx.x;

int x = bx*blockDim.x + tx;
if (x >= nPixels)
return;

weightx[x] = 1.0f / (absIx[x] * absIx[x] + eps);
weighty[x] = 1.0f / (absIy[x] * absIy[x] + eps);
}