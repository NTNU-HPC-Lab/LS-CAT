#include "includes.h"
__global__ void Substep2Kernel (double *Dens, double *VradInt, double *VthetaInt, double *TemperInt, int nrad, int nsec, double *invdiffRmed, double *invdiffRsup, double *DensInt, int Adiabatic, double *Rmed, double dt, double *VradNew, double *VthetaNew, double *Energy, double *EnergyInt)
{
int j = threadIdx.x + blockDim.x*blockIdx.x;
int i = threadIdx.y + blockDim.y*blockIdx.y;

double dv;

if (i<nrad && j<nsec){

dv = VradInt[(i+1)*nsec + j] - VradInt[i*nsec + j];

if (dv < 0.0)
DensInt[i*nsec + j] = CVNR*CVNR*Dens[i*nsec+j]*dv*dv;
else
DensInt[i*nsec + j] = 0.0;

dv = VthetaInt[i*nsec + (j+1)%nsec] - VthetaInt[i*nsec + j];

if (dv < 0.0)
TemperInt[i*nsec + j] = CVNR*CVNR*Dens[i*nsec+j]*dv*dv;
else
TemperInt[i*nsec + j] = 0.0;
}
}