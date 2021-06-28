#include "includes.h"
__global__ void ApplySubKeplerianBoundaryKernel(double *VthetaInt, double *Rmed, double OmegaFrame, int nsec, int nrad, double VKepIn, double VKepOut)
{
int j = threadIdx.x + blockDim.x*blockIdx.x;
int i = 0;

if (j<nsec)
VthetaInt[i*nsec + j] = VKepIn - Rmed[i]*OmegaFrame;

i = nrad - 1;

if (j<nsec)
VthetaInt[i*nsec + j] =  VKepOut - Rmed[i]*OmegaFrame;

}