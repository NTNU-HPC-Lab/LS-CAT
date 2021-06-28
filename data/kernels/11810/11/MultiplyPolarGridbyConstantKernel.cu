#include "includes.h"
__global__ void MultiplyPolarGridbyConstantKernel (double *Dens, int nrad, int nsec, double ScalingFactor)
{
int j = threadIdx.x + blockDim.x*blockIdx.x;
int i = threadIdx.y + blockDim.y*blockIdx.y;

if (i<=nrad && j<nsec)
Dens[i*nsec + j] *= ScalingFactor;
}