#include "includes.h"
__global__ void ComputeConstantResidualKernel (double *VMed, double *invRmed, int *Nshift, int *NoSplitAdvection, int nsec, int nrad, double dt, double *Vtheta, double *VthetaRes, double *Rmed, int FastTransport)
{
int j = threadIdx.x + blockDim.x*blockIdx.x;
int i = threadIdx.y + blockDim.y*blockIdx.y;

double maxfrac, Ntilde, Nround, invdt, dpinvns;
long nitemp;

if (i<nrad && j<nsec){
if (FastTransport)
maxfrac = 1.0;
else
maxfrac = 0.0;

invdt = 1.0/dt;
dpinvns = 2.0*PI/(double)nsec;
Ntilde = VMed[i]*invRmed[i]*dt*(double)nsec/2.0/PI;
Nround = floor(Ntilde+0.5);
nitemp = (long)Nround;
Nshift[i] = (long)nitemp;

Vtheta[i*nsec + j] = (Ntilde-Nround)*Rmed[i]*invdt*dpinvns;
if (maxfrac < 0.5){
NoSplitAdvection[i] = YES;
VthetaRes[i*nsec + j] += Vtheta[i*nsec + j];
Vtheta[i*nsec + j] = 0.0;
}
else{
NoSplitAdvection[i] = NO;
}
}
}