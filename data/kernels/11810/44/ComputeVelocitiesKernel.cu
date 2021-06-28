#include "includes.h"
__global__ void ComputeVelocitiesKernel (double *Vrad, double *Vtheta, double *Dens, double *Rmed, double *ThetaMomP, double *ThetaMomM, double *RadMomP, double *RadMomM, int nrad, int nsec, double OmegaFrame)
{
int j = threadIdx.x + blockDim.x*blockIdx.x;
int i = threadIdx.y + blockDim.y*blockIdx.y;

if (i<nrad && j<nsec){
if (i == 0)
Vrad[i*nsec + j] = 0.0;
else {
Vrad[i*nsec + j] = (RadMomP[(i-1)*nsec + j] + RadMomM[i*nsec + j])/(Dens[i*nsec + j] +
Dens[(i-1)*nsec + j] + 1e-20);
}
Vtheta[i*nsec + j] = (ThetaMomP[i*nsec + ((j-1)+nsec)%nsec] + ThetaMomM[i*nsec + j])/(Dens[i*nsec + j] +
Dens[i*nsec + ((j-1)+nsec)%nsec] + 1e-15)/Rmed[i] - Rmed[i]*OmegaFrame;
/* It was the angular momentum */
}
}