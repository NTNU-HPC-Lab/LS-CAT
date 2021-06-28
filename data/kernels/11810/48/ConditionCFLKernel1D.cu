#include "includes.h"
__global__ void ConditionCFLKernel1D (double *Rsup, double *Rinf, double *Rmed, int nrad, int nsec, double *Vtheta, double *Vmoy)
{
int i = threadIdx.x + blockDim.x*blockIdx.x;
int j;

if (i<nrad){
Vmoy[i] = 0.0;

for (j = 0; j < nsec; j++)
Vmoy[i] += Vtheta[i*nsec + j];

Vmoy[i] /= (double)nsec;
}
}