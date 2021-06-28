#include "includes.h"
__global__ void Compute_weightx_weighty1_norm0_Kernel(float* weightx, float* weighty, const float* psi, const float* phi, const float* absIx, const float* absIy, int nPixels, float norm_for_smooth_term, float eps)
{
int bx = blockIdx.x;
int tx = threadIdx.x;

int x = bx*blockDim.x + tx;
if (x >= nPixels)
return;

weightx[x] = psi[x] / (absIx[x] * absIx[x] + eps);
weighty[x] = phi[x] / (absIy[x] * absIy[x] + eps);
}