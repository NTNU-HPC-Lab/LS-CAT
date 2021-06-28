#include "includes.h"
__global__ void ComputeAverageThetaVelocitiesKernel(double *Vtheta, double *VMed, int nsec, int nrad)
{
int i = threadIdx.x + blockDim.x*blockIdx.x;

double moy = 0.0;
if (i<nrad){
for (int j = 0; j < nsec; j++)
moy += Vtheta[i*nsec + j];

VMed[i] = moy/(double)nsec;
}
}