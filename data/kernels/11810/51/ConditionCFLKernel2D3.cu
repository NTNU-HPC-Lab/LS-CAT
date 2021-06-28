#include "includes.h"
__global__ void ConditionCFLKernel2D3 (double *newDT, double *DT2D, double *DT1D, double *Vmoy, double *invRmed, int *CFL, int nsec, int nrad, double DeltaT)
{
int j = threadIdx.x + blockDim.x*blockIdx.x;

double newdt;
if (j == 0){
newdt = newDT[1];
for (int i=2; i<nrad; i++){
if (newDT[i] < newdt)
newdt = newDT[i];
}

for (int i = 0; i < nrad-1; i++) {
if (DT1D[i] < newdt)
newdt = DT1D[i];
}

if (DeltaT < newdt)
newdt = DeltaT;
CFL[0] = (int)(ceil(DeltaT/newdt));
}
}