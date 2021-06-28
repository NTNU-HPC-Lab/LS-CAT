#include "includes.h"
__global__ void Compute_psi_phi_Kernel(float* psi, float* phi, const float* gAbsIx, const float* gAbsIy, const float* gIx, const float* gIy, int nPixels, float norm_for_contrast_num, float norm_for_contrast_denom, float eps)
{
int bx = blockIdx.x;
int tx = threadIdx.x;

int x = bx*blockDim.x + tx;
if (x >= nPixels)
return;


float psi_num = 0, psi_denom = 0;
float phi_num = 0, phi_denom = 0;
if (norm_for_contrast_num == 0)
{
psi_num = 1;
phi_num = 1;
}
else if (norm_for_contrast_num == 1)
{
psi_num = gAbsIx[x];
phi_num = gAbsIy[x];
}
else if (norm_for_contrast_num == 2)
{
psi_num = gAbsIx[x] * gAbsIx[x];
phi_num = gAbsIy[x] * gAbsIy[x];
}
else
{
psi_num = pow(gAbsIx[x], norm_for_contrast_num);
phi_num = pow(gAbsIy[x], norm_for_contrast_num);
}

if (norm_for_contrast_denom == 0)
{
psi_denom = 1;
phi_denom = 1;
}
else if (norm_for_contrast_denom == 1)
{
psi_denom = fabs(gIx[x]) + eps;
phi_denom = fabs(gIy[x]) + eps;
}
else if (norm_for_contrast_denom == 2)
{
psi_denom = gIx[x] * gIx[x] + eps;
phi_denom = gIy[x] * gIy[x] + eps;
}
else
{
psi_denom = pow(fabs(gIx[x]), norm_for_contrast_denom) + eps;
phi_denom = pow(fabs(gIy[x]), norm_for_contrast_denom) + eps;
}
psi[x] = psi_num / psi_denom;
phi[x] = phi_num / phi_denom;

}