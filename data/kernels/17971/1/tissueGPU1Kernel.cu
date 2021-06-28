#include "includes.h"
/***********************************************************
tissueGPU1.cu
GPU kernel to accumulate contributions of tissue source
strengths qt to tissue solute levels pt.
TWS December 2011
Cuda 10.1 Version, August 2019
************************************************************/


__global__ void tissueGPU1Kernel(int *d_tisspoints, float *d_dtt000, float *d_pt000, float *d_qt000, int nnt)
{
int itp = blockDim.x * blockIdx.x + threadIdx.x;
int jtp,ixyz,ix,iy,iz,jx,jy,jz,nnt2=2*nnt;
float p = 0.;
if(itp < nnt){
ix = d_tisspoints[itp];
iy = d_tisspoints[itp+nnt];
iz = d_tisspoints[itp+nnt2];
for(jtp=0; jtp<nnt; jtp++){
jx = d_tisspoints[jtp];
jy = d_tisspoints[jtp+nnt];
jz = d_tisspoints[jtp+nnt2];
ixyz = abs(jx-ix) + abs(jy-iy) + abs(jz-iz);
p += d_qt000[jtp]*d_dtt000[ixyz];
}
d_pt000[itp] = p;
}
}