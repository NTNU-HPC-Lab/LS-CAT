#include "includes.h"
__global__ void Compute_weightx_weighty1_Kernel(float* weightx, float* weighty, const float* psi, const float* phi, const float* absIx, const float* absIy, int nPixels, float norm_for_smooth_term, float eps)
{
int bx = blockIdx.x;
int tx = threadIdx.x;

int x = bx*blockDim.x + tx;
if (x >= nPixels)
return;
if (norm_for_smooth_term == 2)
{
weightx[x] = psi[x];
weighty[x] = phi[x];
}
else if (norm_for_smooth_term == 1)
{
weightx[x] = psi[x] / (absIx[x] + eps);
weighty[x] = phi[x] / (absIy[x] + eps);
}
else if (norm_for_smooth_term == 0)
{
weightx[x] = psi[x] / (absIx[x] * absIx[x] + eps);
weighty[x] = phi[x] / (absIy[x] * absIy[x] + eps);
}
else
{
weightx[x] = psi[x] / (pow(absIx[x], 2.0f - norm_for_smooth_term) + eps);
weighty[x] = phi[x] / (pow(absIy[x], 2.0f - norm_for_smooth_term) + eps);
}
}