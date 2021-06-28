#include "includes.h"
__global__ void ComputePressureFieldKernel (double *SoundSpeed, double *Dens, double *Pressure, int Adiabatic, int nrad, int nsec, double ADIABATICINDEX, double *Energy) /* LISTO */
{
int j = threadIdx.x + blockDim.x*blockIdx.x;
int i = threadIdx.y + blockDim.y*blockIdx.y;

if (i<nrad && j<nsec){
if (!Adiabatic)
Pressure[i*nsec + j] = Dens[i*nsec + j]*SoundSpeed[i*nsec + j]*SoundSpeed[i*nsec + j];

/* Since SoundSpeed is not update from initialization, cs remains axisymmetric*/
else Pressure[i*nsec + j] = (ADIABATICINDEX-1.0)*Energy[i*nsec + j];
}
}