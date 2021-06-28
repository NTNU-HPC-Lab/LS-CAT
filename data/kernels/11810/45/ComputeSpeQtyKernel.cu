#include "includes.h"
__global__ void ComputeSpeQtyKernel (double *Label, double *Dens, double *ExtLabel, int nrad, int nsec)
{
int j = threadIdx.x + blockDim.x*blockIdx.x;
int i = threadIdx.y + blockDim.y*blockIdx.y;

if (i<nrad && j<nsec){
Label[i*nsec + j] = ExtLabel[i*nsec + j]/Dens[i*nsec + j];
/* Compressive flow if line commentarized
Label[i*nsec + j] = ExtLabel[i*nsec + j] */
}
}