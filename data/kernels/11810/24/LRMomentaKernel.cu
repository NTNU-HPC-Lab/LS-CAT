#include "includes.h"
__global__ void LRMomentaKernel (double *RadMomP, double *RadMomM, double *ThetaMomP, double *ThetaMomM, double *Dens, double *Vrad, double *Vtheta, int nrad, int nsec, double *Rmed, double OmegaFrame)
{
int j = threadIdx.x + blockDim.x*blockIdx.x;
int i = threadIdx.y + blockDim.y*blockIdx.y;

if (i<nrad && j<nsec){
RadMomP[i*nsec + j] = Dens[i*nsec + j] * Vrad[(i+1)*nsec + j]; // (i+1)*nsec
RadMomM[i*nsec + j] = Dens[i*nsec + j] * Vrad[i*nsec + j];
/* it is the angular momentum -> ThetaMomP */
ThetaMomP[i*nsec + j] = Dens[i*nsec + j] * (Vtheta[i*nsec + (j+1)%nsec]+Rmed[i]*OmegaFrame)*Rmed[i];
ThetaMomM[i*nsec + j] = Dens[i*nsec + j] * (Vtheta[i*nsec + j]+Rmed[i]*OmegaFrame)*Rmed[i];
}
}