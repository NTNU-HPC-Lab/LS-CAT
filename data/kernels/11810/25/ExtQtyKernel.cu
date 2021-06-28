#include "includes.h"
__global__ void ExtQtyKernel (double *ExtLabel, double *Dens, double *Label, int nsec, int nrad)
{
int j = threadIdx.x + blockDim.x*blockIdx.x;
int i = threadIdx.y + blockDim.y*blockIdx.y;

if (i<nrad && j<nsec)
ExtLabel[i*nsec + j] = Dens[i*nsec + j]*Label[i*nsec + j];
}