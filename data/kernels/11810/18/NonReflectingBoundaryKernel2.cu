#include "includes.h"
__global__ void NonReflectingBoundaryKernel2 (double *Dens, double *Energy, int i_angle, int nsec, double *Vrad, double *SoundSpeed, double SigmaMed, int nrad, double SigmaMed2, int i_angle2)
{
int j = threadIdx.x + blockDim.x*blockIdx.x;
int i = 1;
double Vrad_med;

Vrad_med = -SoundSpeed[i*nsec + j]*(Dens[i*nsec + j]-SigmaMed)/SigmaMed;
Vrad[i*nsec + j] = 2.0*Vrad_med-Vrad[(i+1)*nsec + j];
i = nrad-1;

Vrad_med = SoundSpeed[i*nsec + j]*(Dens[(i-1)*nsec + j]-SigmaMed2)/SigmaMed2;
Vrad[i*nsec + j] = 2.*Vrad_med - Vrad[(i-1)*nsec + j];

}