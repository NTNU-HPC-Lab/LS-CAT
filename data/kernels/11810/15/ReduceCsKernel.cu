#include "includes.h"
__global__ void ReduceCsKernel (double *SoundSpeed, double *cs0, double *cs1, double *csnrm1, double *csnrm2, int nsec, int nrad)
{
int j = threadIdx.x + blockDim.x*blockIdx.x;
int i=0;

if(j<nsec){
cs0[j] = SoundSpeed[i*nsec +j];
cs1[j] = SoundSpeed[(i+1)*nsec +j];
}
i = nrad-1;
if(j<nsec){
csnrm2[j] = SoundSpeed[(i-1)*nsec +j];
csnrm1[j] = SoundSpeed[i*nsec +j];
}
}