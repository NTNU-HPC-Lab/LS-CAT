#include "includes.h"
__global__ void DivisePolarGridKernel (double *Qbase, double *DensInt, double *Work, int nrad, int nsec)
{
int i = threadIdx.x + blockDim.x*blockIdx.x; //512
int j = threadIdx.y + blockDim.y*blockIdx.y; //256

if (i<=nsec && j<nrad)
Work[i*nrad + j] = Qbase[i*nrad + j]/(DensInt[i*nrad + j] + 1e-20);
}