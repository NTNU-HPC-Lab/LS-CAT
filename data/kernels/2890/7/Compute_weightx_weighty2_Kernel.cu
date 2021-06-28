#include "includes.h"
__global__ void Compute_weightx_weighty2_Kernel(float* weightx, float* weighty, const float* absIx, const float* absIy, int nPixels, float norm_for_smooth_term, float eps)
{
int bx = blockIdx.x;
int tx = threadIdx.x;

int x = bx*blockDim.x + tx;
if (x >= nPixels)
return;
if (norm_for_smooth_term == 2)
{
weightx[x] = 1.0f;
weighty[x] = 1.0f;
}
else if (norm_for_smooth_term == 1)
{
weightx[x] = 1.0f / (absIx[x] + eps);
weighty[x] = 1.0f / (absIy[x] + eps);
}
else if (norm_for_smooth_term == 0)
{
weightx[x] = 1.0f / (absIx[x] * absIx[x] + eps);
weighty[x] = 1.0f / (absIy[x] * absIy[x] + eps);
}
else
{
weightx[x] = 1.0f / (pow(absIx[x], 2.0f - norm_for_smooth_term) + eps);
weighty[x] = 1.0f / (pow(absIy[x], 2.0f - norm_for_smooth_term) + eps);
}
}