#include "includes.h"
__global__ void ConditionCFLKernel2D2 (double *newDT, double *DT2D, double *DT1D, double *Vmoy, double *invRmed, int *CFL, int nsec, int nrad, double DeltaT)
{
int i = threadIdx.x + blockDim.x*blockIdx.x;
int k;
double dt;
double newdt = 1e30;

if (i>0 && i<nrad){
newDT[i] = newdt;
for (k = 0; k < nsec; k++)
if (DT2D[i*nsec + k] < newDT[i]) newDT[i] = DT2D[i*nsec + k]; // for each dt in nrad
}

if (i<nrad-1){
dt = 2.0*PI*CFLSECURITY/(double)nsec/fabs(Vmoy[i]*invRmed[i]-Vmoy[i+1]*invRmed[i+1]);
DT1D[i] = dt; // array nrad size dt
}
}