#include "includes.h"
__global__ void ReduceMeanKernel (double *Dens, double *Energy, int nsec, double *mean_dens, double *mean_energy, double *mean_dens2, double *mean_energy2, int nrad)
{
int j = threadIdx.x + blockDim.x*blockIdx.x;
int i = 0;

if(j<nsec){
mean_dens[j] = Dens[i*nsec+ j];
mean_energy[j] = Energy[i*nsec +j];
}
i = nrad-1;
if(j<nsec){
mean_dens2[j] = Dens[i*nsec + j];
mean_energy2[j] = Energy[i*nsec + j];
}
}