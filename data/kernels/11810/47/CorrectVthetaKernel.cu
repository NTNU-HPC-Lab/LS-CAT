#include "includes.h"
__global__ void CorrectVthetaKernel (double *Vtheta, double domega, double *Rmed, int nrad, int nsec)
{
int j = threadIdx.x + blockDim.x*blockIdx.x;
int i = threadIdx.y + blockDim.y*blockIdx.y;

if (i<nrad && j<nsec)
Vtheta[i*nsec + j] = Vtheta[i*nsec + j] - domega*Rmed[i];
}