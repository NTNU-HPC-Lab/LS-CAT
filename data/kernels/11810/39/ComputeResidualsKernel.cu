#include "includes.h"
__global__ void ComputeResidualsKernel (double *VthetaRes, double *VMed, int nsec, int nrad, double *Vtheta)
{
int j = threadIdx.x + blockDim.x*blockIdx.x;
int i = threadIdx.y + blockDim.y*blockIdx.y;

if (i<nrad && j<nsec)
VthetaRes[i*nsec + j] = Vtheta[i*nsec + j]-VMed[i];
}