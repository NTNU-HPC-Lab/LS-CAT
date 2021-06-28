#include "includes.h"
__global__ void MinusMeanKernel (double *Dens, double *Energy, double SigmaMed, double mean_dens_r, double mean_dens_r2, double mean_energy_r,double mean_energy_r2, double EnergyMed, int nsec, int nrad, double SigmaMed2, double EnergyMed2)
{
int j = threadIdx.x + blockDim.x*blockIdx.x;
int i = 0;
if (j< nsec){
Dens[i*nsec + j] += SigmaMed - mean_dens_r;
Energy[i*nsec + j] += EnergyMed - mean_energy_r;
}

i = nrad-1;
if (j < nsec){
Dens[i*nsec + j] += SigmaMed2 - mean_dens_r2;
Energy[i*nsec + j] += EnergyMed2 - mean_energy_r2;
}
}