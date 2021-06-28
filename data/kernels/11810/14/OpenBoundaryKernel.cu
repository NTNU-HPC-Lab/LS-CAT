#include "includes.h"
__global__ void OpenBoundaryKernel (double *Vrad, double *Dens, double *Energy, int nsec, double SigmaMed)
{
int j = threadIdx.x + blockDim.x*blockIdx.x;
int i = 1;

if(j < nsec){
Dens[(i-1)*nsec + j] = Dens[i*nsec + j]; // copy first ring into ghost ring
Energy[(i-1)*nsec + j] = Energy[i*nsec + j];
if (Vrad[(i+1)*nsec + j] > 0.0 || (Dens[i*nsec + j] < SigmaMed))
Vrad[i*nsec + j] = 0.0; // we just allow outflow [inwards]
else
Vrad[i*nsec +j] = Vrad[(i+1)*nsec + j];
}
}